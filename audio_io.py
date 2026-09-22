"""다온이 음성 입출력 — Python 3.10+

설치:
  python -m pip install sounddevice soundfile webrtcvad-wheels
  Ubuntu에서 PortAudio 오류가 나면: sudo apt install libportaudio2
실행:
  python audio_io.py devices
  python audio_io.py listen                 # 발화 감지/저장만
  python audio_io.py listen --echo          # 녹음 재생 테스트
  python audio_io.py listen --handler my_handler:handle --timeout 45
  python audio_io.py listen --voice-barge-in # 실험용. 아래 제한 필독
  python audio_io.py selftest               # 하드웨어 없는 오류 복구 테스트

listen 중 Enter: 응답 생성/재생 중단. q + Enter 또는 Ctrl+C: 종료.
다른 코드에서는 AudioIO.interrupt() / AudioIO.stop()을 버튼에 연결 가능.

팀원 연결: 같은 폴더의 my_handler.py에 모듈 최상위 함수를 정의한다.
  def handle(input_path: str) -> str | None:
      # STT -> AI -> TTS
      # 입력 WAV의 절대 경로를 받는다.
      # 로컬에 완전히 저장하고 닫은 응답 음성 파일의 절대 경로를 반환.
      # 응답하지 않으면 None. 파일은 재생 완료 전 삭제하지 않는다.
별도 프로세스이므로 부모의 객체/상태 변경은 공유되지 않는다.
람다/중첩 함수 대신 '모듈:함수' 이름을 사용한다. async 함수는 지원 안 함.
핸들러 모듈은 import 때 프로그램을 실행하지 않도록 main 가드를 사용한다.
모델은 자식 프로세스에서 지연 초기화하여 재사용할 수 있다.
시간 초과/취소 시 자식이 종료되므로 다음 요청에서 모델 재초기화가 필요하다.

범위/제한:
- 마이크는 응답 생성/재생 중에도 열려 있다. 입력 큐 크기는 제한한다.
- 기본은 재생 중 음성 판정을 억제한다. Enter/버튼 중단은 가능하다.
- --voice-barge-in은 AEC(음향 에코 제거)가 없는 실험 기능이다.
  스피커 자기 음성/TV도 사용자로 오인한다. 헤드폰/AEC 입력에서 시험할 것.
- 새 발화 시작 시 기존 응답 작업을 취소하는 '최신 발화 우선' 방식이다.
- 오버플로우는 해당 발화를 버리고 복구한다. 입력 장치 오류는 재연결한다.
- STT/AI/TTS 오류는 한 요청만 실패. timeout은 해당 프로세스를 종료한다.
- 강제 종료는 원격 서버 작업/이미 저장한 DB 변경까지 되돌리지 않는다.
  핸들러에서 자손 프로세스를 만들지 말고 외부 요청에도 타임아웃을 걸 것.
- 드라이버/OS 자체 hang, 정전, 디스크 장애까지 복구를 보장하지 않는다.
- 각 발화 최대 30초. 초과 시 분할되며 모델 정확도/화자 구분은 보장 안 됨.
- 기본 recordings 폴더의 사용자 WAV는 작업 종료/취소 후 자동 삭제한다.
  --keep-recordings일 때만 보존한다. TTS 응답 파일은 핸들러가 관리한다.
- 실제 장치의 동시 입출력/음향 테스트가 별도로 필요하다.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import logging
import math
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

LOG = logging.getLogger("daon.audio")


@dataclass
class Config:
    input_device: int | None = None
    output_device: int | None = None
    silence: float = 1.5
    timeout: float = 45.0
    vad_mode: int = 2
    voice_barge_in: bool = False
    keep_recordings: bool = False
    output_dir: str = "recordings"
    rate: int = 16000
    frame_ms: int = 30
    max_seconds: float = 30.0
    min_speech: float = 0.24
    cooldown: float = 0.35

    def __post_init__(self):
        if self.rate not in (8000, 16000, 32000, 48000):
            raise ValueError("지원하지 않는 샘플링 주파수")
        if self.frame_ms not in (10, 20, 30) or self.vad_mode not in range(4):
            raise ValueError("VAD 설정 오류")
        for name in ("silence", "timeout", "max_seconds", "min_speech"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name}은 유한한 양수여야 합니다")
        if self.max_seconds <= self.silence + 0.6:
            raise ValueError("최대 발화 시간이 너무 짧습니다")
        if self.min_speech >= self.max_seconds:
            raise ValueError("최소 음성 시간이 너무 깁니다")
        if not math.isfinite(self.cooldown) or self.cooldown < 0:
            raise ValueError("cooldown은 0 이상이어야 합니다")


class Segmenter:
    """프레임+VAD 판정을 받아 ('start', None), ('utterance', bytes)를 반환."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.pre = deque(maxlen=math.ceil(600 / cfg.frame_ms))
        self.votes = deque(maxlen=math.ceil(300 / cfg.frame_ms))
        self.reset()

    def reset(self):
        self.pre.clear()
        self.votes.clear()
        self.frames = []
        self.active = False
        self.speech = 0
        self.silent = 0

    def feed(self, frame: bytes, voiced: bool):
        c = self.cfg
        if not self.active:
            self.pre.append((frame, voiced))
            self.votes.append(voiced)
            if len(self.votes) == self.votes.maxlen and sum(self.votes) >= math.ceil(len(self.votes) * 0.6):
                self.active = True
                self.frames = [f for f, _ in self.pre]
                self.speech = sum(v for _, v in self.pre)
                self.silent = 0
                for v in reversed(self.votes):
                    if v:
                        break
                    self.silent += 1
                return "start", None
            return None

        self.frames.append(frame)
        self.speech += int(voiced)
        self.silent = 0 if voiced else self.silent + 1
        ended = self.silent * c.frame_ms >= c.silence * 1000
        full = len(self.frames) * c.frame_ms >= c.max_seconds * 1000
        if ended or full:
            data = b"".join(self.frames) if self.speech * c.frame_ms >= c.min_speech * 1000 else None
            self.reset()
            return ("utterance", data) if data else ("discard", None)
        return None


class Microphone:
    """오디오 콜백은 복사/큐 삽입만. 큐 포화/입력 유실은 세대 번호로 전달."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.frames = queue.Queue(maxsize=40)
        self.stop_event = threading.Event()
        self.generation = 0
        self.last_received = 0.0
        self.thread = None

    def _callback(self, indata, frames, timing, status):
        self.last_received = time.monotonic()
        if status:
            self.generation += 1
        item = (self.generation, time.monotonic(), bytes(indata))
        try:
            self.frames.put_nowait(item)
        except queue.Full:
            # 버퍼를 무한히 늘리지 않는다. 다음 프레임의 세대가 바뀐다.
            self.generation += 1

    def start(self):
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()

    def _capture(self):
        import sounddevice as sd
        delay = 0.5
        while not self.stop_event.is_set():
            started = time.monotonic()
            try:
                self.generation += 1
                self.last_received = time.monotonic()
                with sd.RawInputStream(
                    device=self.cfg.input_device,
                    samplerate=self.cfg.rate,
                    channels=1,
                    dtype="int16",
                    blocksize=self.cfg.rate * self.cfg.frame_ms // 1000,
                    callback=self._callback,
                ) as stream:
                    LOG.info("마이크 연결됨")
                    while not self.stop_event.wait(0.1):
                        if not stream.active:
                            raise RuntimeError("입력 스트림 중단")
                        if time.monotonic() - self.last_received > 3:
                            raise RuntimeError("3초 동안 마이크 데이터 수신 없음")
                        if time.monotonic() - started > 5:
                            delay = 0.5
                return
            except Exception as exc:
                LOG.warning("마이크 오류: %s; %.1f초 후 재연결", exc, delay)
                self.generation += 1
                if self.stop_event.wait(delay):
                    return
                delay = min(delay * 2, 8.0)

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                LOG.error("입력 드라이버가 종료에 응답하지 않습니다")


def _worker_main(conn, spec):
    """부모와 독립된 핸들러 프로세스. 정상 요청 사이에는 모델 유지."""
    handler = None
    try:
        while True:
            job_id, path = conn.recv()
            try:
                if handler is None:
                    if spec == "echo":
                        handler = lambda p: p
                    elif spec == "record":
                        handler = lambda p: None
                    elif spec == "__selftest__":
                        handler = _test_handler
                    else:
                        module, name = spec.split(":", 1)
                        handler = getattr(importlib.import_module(module), name)
                    if not callable(handler) or inspect.iscoroutinefunction(handler):
                        handler = None
                        raise TypeError("모듈 최상위 동기 함수를 지정하세요")
                result = handler(path)
                if inspect.isawaitable(result):
                    if inspect.iscoroutine(result):
                        result.close()
                    raise TypeError("async 핸들러는 지원하지 않습니다")
                if result is not None:
                    if not isinstance(result, (str, os.PathLike)):
                        raise TypeError("반환값은 파일 경로 또는 None이어야 합니다")
                    result = str(Path(result).expanduser().resolve())
                    if len(result) > 4096:
                        raise ValueError("반환 경로가 너무 깁니다")
                conn.send((job_id, "ok", result))
            except Exception as exc:
                conn.send((job_id, "error", f"{type(exc).__name__}: {exc}"[:1500]))
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        conn.close()


class HandlerWorker:
    """요청은 하나만. 취소/타임아웃은 프로세스+통신 채널 자체를 폐기."""
    def __init__(self, spec: str, timeout: float):
        self.spec = spec
        self.timeout = timeout
        self.ctx = mp.get_context("spawn")
        self.process = None
        self.conn = None
        self.job = None
        self.deadline = 0.0

    def cancel(self):
        self.job = None
        if self.conn is not None:
            self.conn.close()
            self.conn = None
        if self.process is not None:
            p = self.process
            if p.is_alive():
                p.terminate()
            p.join(timeout=0.2)
            if p.is_alive():
                p.kill()
                p.join(timeout=0.5)
            if p.is_alive():
                # 살아 있는 작업을 놔두고 새 작업을 만들지 않는다.
                raise RuntimeError("작업 프로세스 종료 실패: OS 상태 확인 필요")
            p.close()
            self.process = None

    def submit(self, path: Path):
        if self.job is not None:
            self.cancel()
        if self.process is None or not self.process.is_alive():
            self.cancel()
            parent, child = self.ctx.Pipe()
            self.process = self.ctx.Process(
                target=_worker_main, args=(child, self.spec), daemon=True
            )
            try:
                self.process.start()
            except BaseException:
                parent.close()
                child.close()
                self.process = None
                raise
            child.close()
            self.conn = parent
        self.job = uuid4().hex
        self.deadline = time.monotonic() + self.timeout
        try:
            self.conn.send((self.job, str(path)))
        except Exception:
            self.cancel()
            raise

    def poll(self):
        if self.job is None:
            return None
        if time.monotonic() >= self.deadline:
            self.cancel()
            return "timeout", "응답 처리 제한 시간 초과"
        try:
            if self.conn.poll():
                job_id, status, value = self.conn.recv()
                if job_id != self.job:
                    return None
                self.job = None
                return status, value
            if not self.process.is_alive():
                self.cancel()
                return "error", "응답 처리 프로세스가 비정상 종료됨"
        except (EOFError, BrokenPipeError, OSError):
            self.cancel()
            return "error", "응답 처리 프로세스 연결 끊김"
        return None


class Speaker:
    """독립 출력 스트림. 입력 스트림을 중단하는 sd.stop()은 사용 안 함."""
    def __init__(self, device=None):
        self.device = device
        self.stream = None
        self.done = threading.Event()
        self.error = None
        self.deadline = 0.0

    @property
    def playing(self):
        return self.stream is not None

    def play_file(self, path):
        import sounddevice as sd
        import soundfile as sf
        self.stop()
        # 비정상적으로 큰 응답 파일로 메모리를 소진하지 않도록 제한.
        with sf.SoundFile(str(path)) as file:
            if not 0 < file.frames <= file.samplerate * 120:
                raise ValueError("응답 음성 길이는 0초 초과~120초 이하여야 합니다")
            if file.channels not in (1, 2) or not 8000 <= file.samplerate <= 96000:
                raise ValueError("지원 범위 밖의 음성 형식")
            rate = file.samplerate
            audio = file.read(dtype="float32", always_2d=True)
        if not len(audio):
            raise ValueError("빈 음성 파일")
        position = 0
        self.error = None
        self.done.clear()

        def callback(outdata, frames, timing, status):
            nonlocal position
            outdata.fill(0)
            try:
                if status:
                    self.error = f"출력 데이터 유실: {status}"
                    raise sd.CallbackAbort
                count = min(frames, len(audio) - position)
                outdata[:count] = audio[position:position + count]
                position += count
                if position >= len(audio):
                    raise sd.CallbackStop
            except (sd.CallbackStop, sd.CallbackAbort):
                raise
            except Exception as exc:
                self.error = str(exc)
                self.done.set()
                raise sd.CallbackAbort

        try:
            self.stream = sd.OutputStream(
                samplerate=rate, channels=audio.shape[1], dtype="float32",
                device=self.device, callback=callback,
                finished_callback=self.done.set,
            )
            self.deadline = time.monotonic() + len(audio) / rate + 5
            self.stream.start()
        except BaseException:
            self.stop()
            raise

    def poll(self):
        if not self.stream:
            return False
        if self.done.is_set() or not self.stream.active or time.monotonic() > self.deadline:
            if time.monotonic() > self.deadline:
                self.error = "재생 완료 시간 초과"
            if self.error:
                LOG.warning("재생 실패: %s", self.error)
            self.stop()
            return True
        return False

    def stop(self):
        stream, self.stream = self.stream, None
        if stream is not None:
            try:
                stream.abort()
            finally:
                stream.close()


class AudioIO:
    def __init__(self, cfg: Config, handler="record"):
        self.cfg = cfg
        self.mic = Microphone(cfg)
        self.speaker = Speaker(cfg.output_device)
        self.worker = HandlerWorker(handler, cfg.timeout)
        self.segment = Segmenter(cfg)
        self.stop_event = threading.Event()
        self.interrupt_event = threading.Event()
        self.current_input = None
        self.block_until = 0.0
        self.generation = None

    def interrupt(self):
        # UI/버튼 스레드는 이벤트만 설정. 장치/프로세스 제어는 run에서.
        self.interrupt_event.set()

    def stop(self):
        self.stop_event.set()

    def _cleanup_input(self):
        if self.current_input and not self.cfg.keep_recordings:
            try:
                self.current_input.unlink(missing_ok=True)
            except OSError as exc:
                LOG.warning("임시 녹음 삭제 실패: %s", exc)
        self.current_input = None

    def _cancel(self, preserve_idle_worker=False):
        if not preserve_idle_worker or self.worker.job is not None:
            self.worker.cancel()
        try:
            self.speaker.stop()
        except Exception as exc:
            LOG.warning("스피커 정리 오류: %s", exc)
        self._cleanup_input()

    def _save(self, data):
        folder = Path(self.cfg.output_dir).expanduser().resolve()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"speech_{uuid4().hex}.wav"
        try:
            with wave.open(str(path), "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(self.cfg.rate)
                f.writeframes(data)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return path

    def _tick(self):
        if self.interrupt_event.is_set():
            self.interrupt_event.clear()
            self._cancel()
            self.segment.reset()
            self.block_until = time.monotonic() + self.cfg.cooldown
            LOG.info("사용자 요청으로 응답 중단")
        try:
            finished = self.speaker.poll()
        except Exception as exc:
            LOG.warning("재생 장치 오류: %s", exc)
            self._cancel()
            finished = True
        if finished:
            self._cleanup_input()
            self.segment.reset()
            self.block_until = time.monotonic() + self.cfg.cooldown
        result = self.worker.poll()
        if result is None:
            return
        status, value = result
        if status == "ok" and value is not None:
            try:
                self.speaker.play_file(value)
                self.segment.reset()
            except Exception as exc:
                LOG.warning("응답 재생 실패: %s", exc)
                self._cleanup_input()
        else:
            if status != "ok":
                LOG.warning("이번 응답만 건너뜀 [%s]: %s", status, value)
            self._cleanup_input()

    def run(self):
        import webrtcvad
        vad = webrtcvad.Vad(self.cfg.vad_mode)
        self.mic.start()
        LOG.info("리스닝 시작. Enter: 중단, q+Enter: 종료")
        if self.cfg.voice_barge_in:
            LOG.warning("실험용 음성 바지인 활성: AEC 없으면 자기 음성에 반응할 수 있음")
        last_gap_log = 0.0
        try:
            while not self.stop_event.is_set():
                self._tick()
                try:
                    generation, captured_at, frame = self.mic.frames.get(timeout=0.03)
                except queue.Empty:
                    continue
                now = time.monotonic()
                lost = generation != self.generation
                stale = now - captured_at > 0.5
                if lost or stale:
                    self.segment.reset()
                    vad = webrtcvad.Vad(self.cfg.vad_mode)
                    if self.generation is not None and now - last_gap_log > 2:
                        LOG.warning("입력 유실/지연: 현재 발화를 버리고 다시 감지")
                        last_gap_log = now
                    self.generation = generation
                    continue
                # 재생/쿨다운에 들어온 음성을 나중에 사용자 음성으로 처리하지 않음.
                if captured_at < self.block_until or (
                    self.speaker.playing and not self.cfg.voice_barge_in
                ):
                    self.segment.reset()
                    continue
                try:
                    event = self.segment.feed(frame, vad.is_speech(frame, self.cfg.rate))
                except Exception as exc:
                    LOG.warning("VAD 오류; 현재 발화 초기화: %s", exc)
                    self.segment.reset()
                    vad = webrtcvad.Vad(self.cfg.vad_mode)
                    continue
                if event is None:
                    continue
                kind, data = event
                if kind == "start":
                    # 새 말이 시작되면 이전 작업/응답은 더 이상 재생하지 않는다.
                    self._cancel(preserve_idle_worker=True)
                    LOG.info("발화 시작")
                elif kind == "utterance":
                    try:
                        self.current_input = self._save(data)
                        self.worker.submit(self.current_input)
                        LOG.info("발화 완료; 응답 처리 시작")
                    except Exception as exc:
                        LOG.warning("이번 발화 처리 실패: %s", exc)
                        self._cancel()
        finally:
            self.mic.close()
            self._cancel()


def _test_handler(path):
    name = Path(path).name
    if name == "fail":
        raise ValueError("의도한 실패")
    if name == "hang":
        while True:
            time.sleep(0.1)
    if name == "crash":
        os._exit(7)
    return path


def selftest():
    """VAD 판정은 가짜로 주입; 프로세스 오류/취소는 실제로 실행."""
    cfg = Config(silence=0.3)
    s = Segmenter(cfg)
    events = [s.feed(b"a", True) for _ in range(12)]
    assert any(e and e[0] == "start" for e in events)
    events = [s.feed(b"b", False) for _ in range(10)]
    assert any(e and e[0] == "utterance" for e in events)
    for _ in range(12):
        s.feed(b"a", True)
    s.reset()  # 입력 유실 때 적용하는 동작
    assert not any(s.feed(b"b", False) for _ in range(15))
    for _ in range(12):
        s.feed(b"c", True)
    assert any(s.feed(b"d", False) for _ in range(10))
    LOG.info("PASS: 발화 분리 / 유실 후 재감지")

    mic = Microphone(cfg)
    for _ in range(45):
        mic._callback(b"x", 1, None, False)
    assert mic.frames.qsize() == 40 and mic.generation > 0
    before = mic.generation
    mic._callback(b"x", 1, None, True)
    assert mic.generation > before
    LOG.info("PASS: 입력 오버플로우 / 큐 메모리 상한")

    worker = HandlerWorker("__selftest__", 4)

    def collect():
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            result = worker.poll()
            if result:
                return result
            time.sleep(0.01)
        raise AssertionError("테스트 자체 시간 초과")

    try:
        worker.submit(Path("ok"))
        assert collect()[0] == "ok"
        worker.submit(Path("fail"))
        assert collect()[0] == "error"
        worker.submit(Path("ok"))
        assert collect()[0] == "ok"
        LOG.info("PASS: 콜백 예외 후 다음 요청 성공")
        worker.timeout = 0.3
        worker.submit(Path("hang"))
        assert collect()[0] == "timeout"
        worker.timeout = 4
        worker.submit(Path("ok"))
        assert collect()[0] == "ok"
        LOG.info("PASS: hang 타임아웃/종료 후 재시작")
        worker.submit(Path("hang"))
        worker.cancel()
        worker.submit(Path("ok"))
        assert collect()[0] == "ok"
        assert worker.poll() is None
        LOG.info("PASS: 새 발화 취소 / 오래된 결과 미반영")
        worker.submit(Path("crash"))
        assert collect()[0] == "error"
        worker.submit(Path("ok"))
        assert collect()[0] == "ok"
        LOG.info("PASS: 프로세스 비정상 종료 후 복구")
    finally:
        worker.cancel()
    # 실제 오디오 장치 대신 가짜 출력 장치로 제어 흐름을 검사한다.
    from tempfile import TemporaryDirectory

    class FakeSpeaker:
        def __init__(self):
            self.playing = False
            self.finished = False
            self.last_path = None
        def play_file(self, path):
            self.playing = True
            self.last_path = path
        def poll(self):
            if self.finished:
                self.finished = False
                self.playing = False
                return True
            return False
        def stop(self):
            self.playing = False

    class FakeWorker:
        def __init__(self):
            self.job = None
            self.result = None
            self.cancelled = 0
        def poll(self):
            result, self.result = self.result, None
            return result
        def cancel(self):
            self.job = None
            self.result = None
            self.cancelled += 1

    with TemporaryDirectory() as directory:
        app = AudioIO(Config(output_dir=directory))
        app.worker = FakeWorker()
        app.speaker = FakeSpeaker()
        path = Path(directory) / "input.wav"
        path.write_bytes(b"test")
        app.current_input = path
        app.worker.result = ("ok", str(path))
        app._tick()
        assert app.speaker.playing and path.exists()
        app.speaker.finished = True
        app._tick()
        assert not path.exists() and not app.speaker.playing
        assert app.block_until > time.monotonic()
        app.speaker.playing = True
        app.worker.job = "old"
        app.interrupt()
        app._tick()
        assert not app.speaker.playing and app.worker.job is None
        app._cancel(preserve_idle_worker=True)
        assert app.worker.cancelled == 1
    LOG.info("PASS: 재생 완료/버튼 중단/임시 파일 정리/유휴 모델 유지")
    print("소프트웨어 복구 테스트 통과. 실제 마이크/AEC 검증은 별도입니다.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("devices")
    sub.add_parser("selftest")
    p = sub.add_parser("listen")
    p.add_argument("--input-device", type=int)
    p.add_argument("--output-device", type=int)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--echo", action="store_true")
    group.add_argument("--handler", help="모듈:함수")
    p.add_argument("--timeout", type=float, default=45)
    p.add_argument("--silence", type=float, default=1.5)
    p.add_argument("--vad-mode", type=int, default=2, choices=range(4))
    p.add_argument("--voice-barge-in", action="store_true")
    p.add_argument("--keep-recordings", action="store_true")
    p.add_argument("--output-dir", default="recordings")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.command == "selftest":
        selftest()
        return
    try:
        import sounddevice as sd
        import soundfile
        import webrtcvad
    except (ImportError, OSError) as exc:
        parser.error(f"오디오 의존성 설치/PortAudio 확인 필요: {exc}")
    if args.command == "devices":
        print(sd.query_devices())
        return
    if args.handler and ":" not in args.handler:
        parser.error("--handler는 my_handler:handle 형식이어야 합니다")
    cfg = Config(
        input_device=args.input_device, output_device=args.output_device,
        silence=args.silence, timeout=args.timeout, vad_mode=args.vad_mode,
        voice_barge_in=args.voice_barge_in,
        keep_recordings=args.keep_recordings, output_dir=args.output_dir,
    )
    app = AudioIO(cfg, args.handler or ("echo" if args.echo else "record"))

    def keyboard():
        while not app.stop_event.is_set():
            try:
                command = input().strip().lower()
            except (EOFError, OSError):
                return
            if command == "q":
                app.stop()
                return
            app.interrupt()

    if sys.stdin.isatty():
        threading.Thread(target=keyboard, daemon=True).start()
    try:
        app.run()
    except KeyboardInterrupt:
        LOG.info("종료")


if __name__ == "__main__":
    mp.freeze_support()
    main()
