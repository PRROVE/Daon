"""Run the real pipeline once and print its actual outputs for a terminal recording.

Examples (daon_v2 environment):
    python scripts/demo_story.py --audio recording.wav --nickname "바다 님"
    python scripts/demo_story.py --sample c04 --wait
The recording gates are outside Pipeline.run and its E2E measurement.
"""

import argparse
import json
import os
import sys
import threading
import unicodedata
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESET = "\033[0m\033[1m"
BLUE = "\033[1;96m"
GREEN = "\033[1;92m"
YELLOW = "\033[1;93m"
DIM = "\033[90m"


def wrapped(text: str, columns: int = 84) -> None:
    """Wrap Korean text by terminal columns, without changing its content."""
    line, width = "", 0
    for char in text:
        size = 2 if unicodedata.east_asian_width(char) in "WF" else 1
        if width + size > columns:
            print("  " + line)
            line, width = "", 0
        line += char
        width += size
    if line:
        print("  " + line)


class DemoProgress:
    """Progress mixin; the runtime Pipeline still performs all inference."""

    def __init__(self, settings, out_dir: Path) -> None:
        self.events_path = out_dir / "events.jsonl"
        super().__init__(settings)

    def event(self, name: str, **fields) -> None:
        payload = {"event": name, "at": datetime.now().astimezone().isoformat(), **fields}
        with self.events_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _timed(self, rec, stage, key, fn):
        titles = {
            "stt": "[1/3] 음성 전사 | KoMix V2",
            "llm": "[2/3] 사연 생성 + 후처리 | Qwen3 4B + Kiwi",
            "tts": "[3/3] 음성 합성 | Supertonic · M1 · 한국어",
        }
        print(f"\n{BLUE}{titles[stage]}{RESET}")
        self.event("stage_start", stage=stage)
        started = perf_counter()
        stop = threading.Event()

        def progress():
            marks = ("|", "/", "-", "\\")
            step = 0
            while not stop.is_set():
                print(f"\r  {YELLOW}{marks[step % 4]} 실제 처리 중 · {perf_counter() - started:5.1f}초{RESET}\033[K", end="", flush=True)
                step += 1
                stop.wait(0.2)

        worker = threading.Thread(target=progress, daemon=True)
        worker.start()
        try:
            succeeded = super()._timed(rec, stage, key, fn)
        finally:
            stop.set()
            worker.join()
            print("\r\033[K", end="", flush=True)
        print(f"  {GREEN if succeeded else YELLOW}{'완료' if succeeded else '실패'} · {rec[stage + '_sec']:.1f}초{RESET}")
        self.event("stage_end", stage=stage, seconds=rec[stage + "_sec"], ok=succeeded)
        if succeeded and stage in ("stt", "llm"):
            wrapped(rec[key])
        elif succeeded:
            print(f"  저장: {display_path(Path(rec[key]))}")
        return succeeded


def display_path(path: Path) -> str:
    path = path.resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


@dataclass(frozen=True)
class DemoInput:
    audio_path: Path
    nickname: str
    sample_id: str | None = None


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--audio", type=Path, help="Use your own WAV without a dataset or reference transcript")
    inputs.add_argument("--sample", help="Sample ID in the manifest (default: c04)")
    parser.add_argument("--manifest", type=Path, help="Dataset manifest (default: data/dev_clear/manifest.jsonl)")
    parser.add_argument("--nickname", help="Nickname for the story (default for --audio: 청취자 님)")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--wait", action="store_true", help="Wait for Enter before and after the measured run")
    return parser


def resolve_input(args: argparse.Namespace) -> DemoInput:
    """Validate local input before importing or loading inference models."""
    if args.audio is not None:
        if args.manifest is not None:
            raise ValueError("--audio와 --manifest는 함께 사용할 수 없습니다.")
        audio_path = args.audio.expanduser().resolve()
        nickname = args.nickname if args.nickname is not None else "청취자 님"
        sample_id = None
    else:
        manifest = (args.manifest or ROOT / "data" / "dev_clear" / "manifest.jsonl").expanduser().resolve()
        if not manifest.is_file():
            raise ValueError(f"데이터셋 manifest가 없습니다: {manifest}. 직접 녹음한 WAV는 --audio 파일경로로 실행하세요.")
        rows = []
        for number, line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"manifest {number}행의 JSON이 올바르지 않습니다: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"manifest {number}행은 JSON 객체여야 합니다.")
            rows.append(row)
        sample_id = args.sample or "c04"
        selected = [row for row in rows if row.get("sample_id") == sample_id]
        if len(selected) != 1:
            raise ValueError(f"sample_id={sample_id!r}는 manifest에 정확히 한 번 있어야 합니다. 발견: {len(selected)}건")
        sample = selected[0]
        if not isinstance(sample.get("wav"), str) or not sample["wav"].strip():
            raise ValueError(f"sample_id={sample_id!r}에 wav 경로가 없습니다.")
        audio_path = (manifest.parent / sample["wav"]).resolve()
        nickname = args.nickname if args.nickname is not None else sample.get("nickname")
    if not isinstance(nickname, str) or not nickname.strip():
        raise ValueError("비어 있지 않은 nickname이 필요합니다. --nickname으로 지정하세요.")
    if not audio_path.is_file():
        raise ValueError(f"입력 음성 파일이 없습니다: {audio_path}")
    return DemoInput(audio_path, nickname.strip(), sample_id)


def create_pipeline(settings, out_dir: Path):
    # Keep --help and input validation usable before installing model runtimes.
    from src.pipeline import Pipeline

    class DemoPipeline(DemoProgress, Pipeline):
        pass

    return DemoPipeline(settings, out_dir)


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    try:
        sample = resolve_input(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    # Whisper's audio loader calls ffmpeg by name in a conda environment.
    conda_bin = Path(sys.executable).parent / "Library" / "bin"
    if conda_bin.is_dir():
        os.environ["PATH"] = str(conda_bin) + os.pathsep + os.environ.get("PATH", "")
    warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
    warnings.filterwarnings("ignore", message="Performing inference on CPU when CUDA is available")
    try:
        import soundfile as sf
        from config.settings import Settings
    except ImportError as exc:
        parser.error(f"실행 환경의 패키지를 확인하세요: {exc}. README의 환경 설정을 먼저 완료하세요.")
    try:
        audio = sf.info(str(sample.audio_path))
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(f"입력 음성을 읽을 수 없습니다: {exc}")
    if audio.frames == 0:
        parser.error("입력 음성이 비어 있습니다.")
    out_dir = (args.out_dir or ROOT / "results" / ("demo_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print("실제 모델을 준비합니다. 모델 로딩은 E2E 측정에 포함하지 않습니다.", flush=True)
    settings = Settings()
    pipeline = create_pipeline(settings, out_dir)
    print("\033[2J\033[H", end="")
    print(f"{BLUE}DAON V2 | 실제 음성 사연 생성{RESET}")
    print("─" * 70)
    print(f"입력: {display_path(sample.audio_path)} · {audio.duration:.2f}초    별명: {sample.nickname}")
    print(f"STT: {settings.device.upper()}    LLM: {'CPU' if settings.llm['num_gpu'] == 0 else '설정에 따름'}    모델 로딩 완료")
    print(f"{DIM}실제 Pipeline.run 실행 · 완성된 결과를 그대로 표시합니다.{RESET}")
    if args.wait:
        input("\nEnter를 누르면 음성 전사부터 시작합니다.")
    pipeline.event("run_start", sample_id=sample.sample_id, audio_duration=audio.duration)
    result = pipeline.run(sample.audio_path, sample.nickname, out_dir / "audio")
    pipeline.event("run_end", e2e_sec=result["e2e_sec"], ok=result["ok"])
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "─" * 70)
    if result["ok"]:
        print(f"{GREEN}생성 완료 · E2E {result['e2e_sec']:.1f}초 · 사연 {len(result['story'])}자{RESET}")
        print("입력 음성 → 전사문 → 최종 사연 → story.wav")
    else:
        print(f"실행 실패: {result['failed_stage']} · {result['error']}")
    if args.wait:
        input("\nEnter를 누르면 종료합니다.")
    print("\033[0m", end="", flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
