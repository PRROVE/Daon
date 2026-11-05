import json
import platform
import subprocess
import sys
import threading
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import psutil
import torch

from config.paths import CONFIG_PATH
from config.settings import Settings
from src.pipeline import Pipeline


class PipelineRunner:
    """평가 샘플을 워밍업 1회 + reps 바퀴 순차 실행해 results/{run_id}/pipeline.jsonl 에 요청별 기록을 남긴다."""

    def __init__(self, pipeline: Pipeline, set_name: str, set_dir: Path, results_dir: Path, reps: int) -> None:
        self.pipeline = pipeline
        self.set_name = set_name
        self.set_dir = set_dir
        self.results_dir = results_dir
        self.reps = reps
        self.rows = [json.loads(l) for l in (set_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]

    def run(self) -> Path:
        """run 폴더를 만들고 전부 실행한 뒤 그 경로를 돌려준다. 요청마다 바로 기록해 중간에 죽어도 남는다."""
        now = datetime.now(timezone.utc).astimezone()
        run_dir = self.results_dir / f"{self.set_name}_{now.strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(json.dumps(self._run_info(run_dir, now), ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "config.yaml").write_bytes(CONFIG_PATH.read_bytes())
        peak, stop = {"mib": 0}, threading.Event()
        threading.Thread(target=self._sample_gpu, args=(peak, stop), daemon=True).start()
        try:
            # 평가지표 ③: 워밍업 1건은 집계에서 제외하되 흔적은 남긴다
            self._execute(run_dir, self.rows[0], rep=0, warmup=True)
            # 같은 입력이 연달아 오지 않도록 전체를 reps 바퀴 돈다 (Ollama 프롬프트 캐시가 2·3회차를 유리하게 만들지 않게)
            for rep in range(1, self.reps + 1):
                for row in self.rows:
                    self._execute(run_dir, row, rep=rep, warmup=False)
        finally:
            stop.set()
            info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            info.update({"gpu_peak_mib": peak["mib"], "finished_at": datetime.now(timezone.utc).astimezone().isoformat()})
            (run_dir / "run.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        return run_dir

    @staticmethod
    def _sample_gpu(peak: dict, stop: threading.Event) -> None:
        # GPU 피크는 whisper(이 프로세스)와 Ollama(다른 프로세스)를 합친 값이어야 해서 nvidia-smi 전체 사용량을 2초마다 읽는다
        while not stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5, check=False)
            except (OSError, subprocess.TimeoutExpired):
                return
            if out.returncode == 0 and out.stdout.strip():
                peak["mib"] = max(peak["mib"], int(out.stdout.strip().splitlines()[0]))
            stop.wait(2)

    def _execute(self, run_dir: Path, row: dict, rep: int, warmup: bool) -> None:
        request_id = f"{row['sample_id']}_warmup" if warmup else f"{row['sample_id']}_r{rep}"
        rec = self.pipeline.run(self.set_dir / row["wav"], row["nickname"], run_dir / "audio" / request_id)
        rec.update({
            "request_id": request_id, "rep": rep, "warmup": warmup,
            "sample_id": row["sample_id"], "speaker_id": row["speaker_id"], "topic": row["topic"],
            "length_bucket": row["length_bucket"], "episode_id": row["episode_id"],
            "input_duration": row["duration"],
            "ref_transcript": (self.set_dir / row["transcript"]).read_text(encoding="utf-8"),
        })
        with (run_dir / "pipeline.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        status = "ok" if rec["ok"] else f"FAIL@{rec['failed_stage']}"
        print(f"{request_id:<14} {status:<10} e2e={rec['e2e_sec']:6.1f}s", flush=True)

    def _run_info(self, run_dir: Path, now: datetime) -> dict:
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        return {
            "run_id": run_dir.name, "set_name": self.set_name, "set_dir": str(self.set_dir),
            "n_samples": len(self.rows), "reps": self.reps, "warmup": 1, "started_at": now.isoformat(),
            # 평가지표 ③: 실행 환경(GPU·RAM)을 기록한다. 특정 기기로 고정하지 않는다
            "env": {"os": platform.platform(), "python": platform.python_version(), "gpu": gpu,
                    "ram_gb": round(psutil.virtual_memory().total / 2**30, 1), "torch": torch.__version__,
                    "openai-whisper": version("openai-whisper"), "supertonic": version("supertonic")},
            **self.pipeline.meta,
        }


if __name__ == "__main__":
    set_name = sys.argv[1] if len(sys.argv) > 1 else "eval_set"
    s = Settings()
    runner = PipelineRunner(Pipeline(s), set_name, getattr(s, set_name)["out_dir"], s.eval["results_dir"], s.eval["reps"])
    print("run_dir:", runner.run())
