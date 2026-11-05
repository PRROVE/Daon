import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path

import laya
import torch

from config.settings import Settings


class LayaEvaluator:
    """검증한 로컬 가중치로 원본 정답 전사문과 생성 사연을 비교한다."""

    def __init__(self, config: dict) -> None:
        self.config = config
        if not 0 <= config["binary_threshold"] <= 1:
            raise ValueError("binary_threshold must be in [0,1]")
        model_dir = config["model_dir"]
        hashes = self._verify(model_dir, config)
        self.questions = json.loads(config["questions_path"].read_text(encoding="utf-8"))
        self.questions = {k: v for k, v in self.questions.items() if k not in ("R2", "R6")}
        self.device = config["device"]
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.agent = laya.load(str(model_dir), device=self.device, compile=False)
        self.metadata = self._metadata(hashes)
        self.signature = hashlib.sha256(json.dumps(self.metadata, sort_keys=True).encode()).hexdigest()

    def evaluate(self, state: dict) -> dict:
        """잘린 입력이나 비정상 점수는 거부하고 수치와 판정을 돌려준다."""
        if not all(isinstance(state.get(k), str) and state[k].strip() for k in ("nickname", "reference", "story")):
            raise ValueError("nickname/reference/story must be non-empty; reference must be original label text")
        prediction = self._predict(state)
        answers = prediction["answers"]
        score = self._score(answers["preservation"]["score"], 10)
        binary = {k: self._score(answers[k]["noul"], 1) for k in self.questions if k != "preservation"}
        return {
            "content": self._content(score, binary),
            "format": self._format(binary),
            "raw_prediction": prediction,
        }

    def _predict(self, state: dict) -> dict:
        prediction = self.agent.predict_batch(
            [state], self.questions, batch_size=1,
            max_len=self.config["max_len"], head_max_len=self.config["head_max_len"],
        )[0]
        if prediction["usage"]["truncated"]:
            raise ValueError("Laya input truncated; no judgment accepted")
        return prediction

    def _content(self, score: float, binary: dict) -> dict:
        threshold = self.config["binary_threshold"]
        content = {
            "preservation_score": score, "preservation_ratio": score / 10,
            "preservation_method": "laya_estimate", "all_preserved_estimate": score == 10.0,
        }
        for k in ("distortion", "addition"):
            content[k] = {"score": binary[k], "found": binary[k] >= threshold, "evidence": None}
        return content

    def _format(self, binary: dict) -> dict:
        threshold = self.config["binary_threshold"]
        return {k: {"score": binary[k], "pass": binary[k] >= threshold, "evidence": None}
                for k in ("R1", "R3", "R4", "R5", "R7")}

    def _metadata(self, hashes: dict) -> dict:
        return {
            "model": self.config["model"], **hashes, "device": self.device,
            "gpu": torch.cuda.get_device_name() if self.device.startswith("cuda") else None,
            "laya_version": version("laya"), "torch_version": torch.__version__,
            "binary_threshold": self.config["binary_threshold"], "max_len": self.config["max_len"],
            "head_max_len": self.config["head_max_len"], "schema": "laya-v1",
        }

    @staticmethod
    def _score(value: float, maximum: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Invalid Laya numeric score: {value!r}")
        if not math.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError(f"Laya score outside [0,{maximum}]: {value!r}")
        return float(value)

    @classmethod
    def _verify(cls, model_dir: Path, config: dict) -> dict:
        weights = cls._hash(model_dir / "model.safetensors")
        questions = cls._hash(config["questions_path"])
        if weights != config["weights_sha256"] or questions != config["questions_sha256"]:
            raise ValueError("Laya checkpoint/questions hash mismatch")
        artifacts = {str(p.relative_to(model_dir)): cls._hash(p) for p in sorted(model_dir.rglob("*.json"))
                     if p.name != "train_report.json"}
        return {"weights_sha256": weights, "questions_sha256": questions, "artifact_hashes": artifacts}

    @staticmethod
    def _hash(path: Path) -> str:
        with path.open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()


if __name__ == "__main__":
    cfg = Settings().judge
    evaluator = LayaEvaluator(cfg)
    print(json.dumps(evaluator.metadata, ensure_ascii=False, indent=2))
