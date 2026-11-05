import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from config.settings import Settings
from eval.laya_evaluator import LayaEvaluator


class Judge:
    """원본 라벨과 생성 사연을 Laya로 평가하고 입력·모델을 대조해 재개한다."""

    def __init__(self, evaluator: LayaEvaluator) -> None:
        self.evaluator = evaluator

    def judge(self, run_dir: Path) -> int:
        """미평가 또는 변경된 요청만 판정하고 이번 성공 건수를 돌려준다."""
        if not (run_dir / "pipeline.jsonl").is_file():
            raise FileNotFoundError(run_dir / "pipeline.jsonl")
        records = [r for r in self._read(run_dir / "pipeline.jsonl") if not r["warmup"]]
        path = run_dir / "judgments.jsonl"
        latest = {r["request_id"]: r for r in self._read(path)}
        if any(r.get("schema") != "laya-v1" for r in latest.values()):
            raise ValueError("Historical judgments exist. Copy pipeline.jsonl/run.json to a new run before rejudging.")
        success, skipped = 0, 0
        for rec in records:
            old = latest.get(rec["request_id"], {})
            if self._current(old, rec):
                skipped += 1
                continue
            out = self._judge_one(rec)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
            success += out["error"] is None
            status = "judged" if out["error"] is None else "미평가 " + out["error"][:120]
            print(f"{rec['request_id']:<14} {status}", flush=True)
        print(f"이번 판정 {success}건 / 기존 완료 {skipped}건 / 미평가 {len(records)-success-skipped}건")
        return success

    def _current(self, old: dict, rec: dict) -> bool:
        return bool(old and old.get("error") is None and old.get("input_sha256") == self._input_hash(rec)
                    and old.get("judge_signature") == self.evaluator.signature)

    def _judge_one(self, rec: dict) -> dict:
        out = {"schema": "laya-v1", "request_id": rec["request_id"], "sample_id": rec["sample_id"],
               "judged_at": datetime.now(timezone.utc).astimezone().isoformat(),
               "judge_model": self.evaluator.metadata["model"], "judge_metadata": self.evaluator.metadata,
               "judge_signature": self.evaluator.signature, "input_sha256": self._input_hash(rec),
               "generated": bool((rec.get("story") or "").strip()), "content": None, "format": None, "error": None}
        if not out["generated"]:
            return out
        try:
            out.update(self.evaluator.evaluate(self._state(rec)))
        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            # 개별 추론 실패는 미평가로 남겨 다음 요청을 처리하고 재실행 때 다시 판정한다.
            out["error"] = repr(e)
        return out

    @staticmethod
    def _state(rec: dict) -> dict:
        return {"nickname": rec.get("nickname"), "reference": rec.get("ref_transcript"), "story": rec.get("story")}

    @classmethod
    def _input_hash(cls, rec: dict) -> str:
        return hashlib.sha256(json.dumps(cls._state(rec), ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _read(path: Path) -> list[dict]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m eval.judge <run_id>")
    settings = Settings()
    run_dir = settings.eval["results_dir"] / sys.argv[1]
    evaluator = LayaEvaluator(settings.judge)
    Judge(evaluator).judge(run_dir)
