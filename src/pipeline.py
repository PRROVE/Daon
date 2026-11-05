import hashlib
import json
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from config.paths import DATA_DIR, RESULTS_DIR, ROOT
from config.settings import Settings
from src.formatter import StoryFormatter
from src.llm import LLM
from src.stt import STT
from src.tts import TTS


class Pipeline:
    """STT → LLM → TTS를 순서대로 실행하고 단계별 시간과 결과를 기록한다."""

    def __init__(self, settings: Settings) -> None:
        s, l, t = settings.stt, settings.llm, settings.tts
        self.stt = STT(device=settings.device, **s)
        self.llm = LLM(host=l["host"], model=l["model"], think=l["think"], num_gpu=l["num_gpu"],
                       timeout_sec=l["timeout_sec"], prompt_path=l["prompt_path"])
        self.tts = TTS(voice=t["voice"], lang=t["lang"], model_dir=t["model_dir"])
        self.formatter = StoryFormatter(settings.pipeline["intro"], settings.pipeline["closing"], settings.llm["target_chars"]["max"])
        self.out_wav_name: str = settings.pipeline["out_wav_name"]
        # 평가 중 프롬프트가 바뀌지 않았음을 결과 파일로 증명하기 위해 고정본 해시와 대조한다
        prompt_sha256 = hashlib.sha256(Path(l["prompt_path"]).read_bytes()).hexdigest()
        if l["prompt_sha256"] and l["prompt_sha256"] != prompt_sha256:
            raise ValueError(f"prompts/story.md 가 고정본과 다름: {prompt_sha256} != config {l['prompt_sha256']}")
        # 경로는 ROOT 기준으로 남겨 다른 PC 의 결과와 그대로 비교할 수 있게 한다
        self.meta = {"stt_model": s["model"], "llm_model": l["model"],
                     "prompt_path": Path(l["prompt_path"]).relative_to(ROOT).as_posix(), "prompt_sha256": prompt_sha256}

    def run(self, audio_path: Path, nickname: str, out_dir: Path) -> dict:
        """음성 1건을 끝까지 처리하고 단계별 시간·결과·실패 정보를 dict로 돌려준다."""
        rec: dict = {
            "audio_path": str(audio_path), "nickname": nickname, "out_wav": None,
            "started_at": datetime.now(timezone.utc).astimezone().isoformat(), "ended_at": None,
            "stt_sec": None, "llm_sec": None, "tts_sec": None, "e2e_sec": None,
            "transcript": None, "story": None,
            "ok": False, "failed_stage": None, "error": None, **self.meta,
        }
        t0 = perf_counter()
        stages: list[tuple[str, str, Callable[[], object]]] = [
            ("stt", "transcript", lambda: self.stt.transcribe(audio_path)),
            ("llm", "story", lambda: self.formatter.format(self._clean_story(self.llm.generate(rec["transcript"], nickname)), rec["transcript"], nickname)),
            ("tts", "out_wav", lambda: str(self.tts.synthesize(rec["story"], out_dir / self.out_wav_name))),
        ]
        for stage, key, fn in stages:
            if not self._timed(rec, stage, key, fn):
                break
        else:
            rec["ok"] = True
        rec["e2e_sec"] = perf_counter() - t0
        rec["ended_at"] = datetime.now(timezone.utc).astimezone().isoformat()
        return rec

    @staticmethod
    def _clean_story(text: str) -> str:
        """R2 글자 수와 TTS 입력의 기준이 되는 문자열: 마크다운 기호·줄바꿈을 걷어내고 공백을 정리한다."""
        lines = [re.sub(r"^\s*(?:[-*>#]+|\d+[.)])\s*", "", line) for line in text.splitlines()]
        return " ".join(re.sub(r"[*_`#>]", "", " ".join(lines)).split())

    def _timed(self, rec: dict, stage: str, key: str, fn: Callable[[], object]) -> bool:
        """단계 하나를 시간 재며 실행. 실패하면 사유를 rec에 적고 False."""
        t = perf_counter()
        # 평가지표 ③: 어떤 단계가 어떤 이유로 죽든 재시도 없이 그대로 기록해야 하므로 여기서만 전부 잡는다
        try:
            rec[key] = fn()
        except Exception as e:  # noqa: BLE001
            rec["failed_stage"], rec["error"] = stage, repr(e)
            return False
        finally:
            rec[f"{stage}_sec"] = perf_counter() - t
        return True


if __name__ == "__main__":
    pipeline = Pipeline(Settings())
    result = pipeline.run(DATA_DIR / "dev_set" / "s01.wav", "은행 님", RESULTS_DIR / "_smoke")
    print(json.dumps(result, ensure_ascii=False, indent=2))
