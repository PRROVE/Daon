from pathlib import Path

import yaml

from config.paths import CONFIG_PATH, ROOT


class Settings:
    """config.yaml을 읽어 섹션별 dict로 보관한다."""

    def __init__(self, path: Path = CONFIG_PATH) -> None:
        with path.open(encoding="utf-8") as f:
            raw: dict = yaml.safe_load(f)
        self.path = path
        self.device: str = raw["device"]
        self.stt: dict = raw["stt"]
        self.llm: dict = raw["llm"]
        self.tts: dict = raw["tts"]
        self.pipeline: dict = raw["pipeline"]
        self.eval_set: dict = raw["eval_set"]
        self.dev_set: dict = raw["dev_set"]
        self.dev_clear: dict = raw["dev_clear"]
        self.judge: dict = raw["judge"]
        self.eval: dict = raw["eval"]
        # yaml의 상대 경로는 실행 위치와 무관하게 ROOT 기준이어야 한다
        self.stt["model_dir"] = ROOT / self.stt["model_dir"]
        self.stt["vad_dir"] = ROOT / self.stt["vad_dir"]
        self.llm["prompt_path"] = ROOT / self.llm["prompt_path"]
        self.tts["model_dir"] = ROOT / self.tts["model_dir"]
        for key in ("model_dir", "questions_path"):
            self.judge[key] = ROOT / self.judge[key]
        self.eval["results_dir"] = ROOT / self.eval["results_dir"]
        for section in (self.eval_set, self.dev_set, self.dev_clear):
            section["out_dir"] = ROOT / section["out_dir"]


if __name__ == "__main__":
    s = Settings()
    print(s.path)
    print("device", s.device)
    print("stt", s.stt)
    print("llm", s.llm)
    print("tts", s.tts)
    print("eval", s.eval)
