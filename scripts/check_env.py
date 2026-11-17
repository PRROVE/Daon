"""Check packages, model files and Ollama without running or downloading models."""

import argparse
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EVALUATION_PACKAGES = {"laya", "transformers", "tokenizers", "huggingface-hub", "safetensors"}


@dataclass(frozen=True)
class Check:
    status: str
    name: str
    detail: str


def check_file(name: str, path: Path, expected_hash: str | None = None) -> Check:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return Check("FAIL", name, f"파일이 없거나 비어 있음: {path}")
        if expected_hash is not None:
            with path.open("rb") as file:
                digest = hashlib.file_digest(file, "sha256").hexdigest()
            if digest != expected_hash:
                return Check("FAIL", name, f"SHA256 불일치: {path}. 설정과 일치하는 배포 파일·프롬프트를 확인하세요.")
        return Check("OK", name, f"{'SHA256 일치' if expected_hash is not None else '파일 있음'}: {path}")
    except OSError as exc:
        return Check("FAIL", name, str(exc))


def check_packages(generation_only: bool = False) -> list[Check]:
    checks = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8-sig").splitlines():
        match = re.fullmatch(r"([\w.-]+)==([^\s#]+)", line.strip())
        if not match:
            continue
        name, expected = match.groups()
        if generation_only and name in EVALUATION_PACKAGES:
            continue
        try:
            installed = version(name)
        except PackageNotFoundError:
            checks.append(Check("FAIL", name, "미설치. README의 패키지 설치 단계를 확인하세요."))
            continue
        # The CUDA build of torch has a local suffix such as +cu128.
        matches = installed == expected or (name == "torch" and installed.split("+")[0] == expected)
        checks.append(Check("OK" if matches else "FAIL", name, f"설치 {installed} / 요구 {expected}"))
    return checks


def check_ollama(host: str, model: str) -> Check:
    try:
        with urlopen(host.rstrip("/") + "/api/tags", timeout=3) as response:
            payload = json.load(response)
        if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
            raise ValueError("models 목록이 없는 응답")
        names = {item.get("name") or item.get("model") for item in payload["models"] if isinstance(item, dict)}
        if model not in names and not (":" not in model and model + ":latest" in names):
            return Check("FAIL", "Ollama 모델", f"{model} 없음. README의 ollama pull/cp 단계를 완료하세요.")
        return Check("OK", "Ollama 모델", f"서버 연결 및 {model} 등록 확인")
    except (URLError, OSError, ValueError) as exc:
        return Check("FAIL", "Ollama 서버", f"{host} 확인 실패: {exc}. ollama serve와 모델 경로를 확인하세요.")


def check_models(settings, generation_only: bool = False) -> list[Check]:
    stt, llm, tts = settings.stt, settings.llm, settings.tts
    checks = [
        check_file("사연 프롬프트", llm["prompt_path"], llm["prompt_sha256"]),
        check_file("STT 가중치", stt["model_dir"] / (stt["model"] + ".pt")),
        check_file("Silero VAD 로더", stt["vad_dir"] / "hubconf.py"),
        check_file("Silero VAD 가중치", stt["vad_dir"] / "src/silero_vad/data/silero_vad.jit"),
    ]
    tts_files = [tts["model_dir"] / "onnx" / name for name in (
        "duration_predictor.onnx", "text_encoder.onnx", "vector_estimator.onnx", "vocoder.onnx",
        "tts.json", "unicode_indexer.json",
    )]
    tts_files.append(tts["model_dir"] / "voice_styles" / (tts["voice"] + ".json"))
    missing = [str(path.relative_to(tts["model_dir"])) for path in tts_files if not path.is_file() or path.stat().st_size == 0]
    checks.append(Check("WARN" if missing else "OK", "Supertonic 파일",
                        "첫 실행 시 다운로드·준비 필요: " + ", ".join(missing) if missing else "필요한 모델·음색 파일 있음"))
    if not generation_only:
        judge = settings.judge
        checks += [check_file("Laya 가중치", judge["model_dir"] / "model.safetensors", judge["weights_sha256"]),
                   check_file("Laya 질문", judge["questions_path"], judge["questions_sha256"])]
        for name in ("encoder/config.json", "rl_agent_config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"):
            checks.append(check_file("Laya " + name, judge["model_dir"] / name))
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-only", action="store_true", help="Skip Laya evaluation packages and checkpoint files")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    if sys.version_info < (3, 11):
        print("[FAIL] Python 3.11 이상이 필요합니다. 확인된 환경은 Python 3.11입니다.")
        return 1
    checks = [Check("OK" if sys.version_info[:2] == (3, 11) else "WARN", "Python",
                    f"{sys.version.split()[0]} (확인된 환경: 3.11)")]
    checks += check_packages(args.generation_only)
    conda_ffmpeg = Path(sys.executable).parent / "Library/bin/ffmpeg.exe"
    ffmpeg = shutil.which("ffmpeg") or (str(conda_ffmpeg) if conda_ffmpeg.is_file() else None)
    checks.append(Check("OK" if ffmpeg else "FAIL", "ffmpeg", ffmpeg or "찾을 수 없음. README의 ffmpeg 설치 단계를 확인하세요."))
    try:
        from config.settings import Settings

        settings = Settings()
        checks += check_models(settings, args.generation_only)
        checks.append(check_ollama(settings.llm["host"], settings.llm["model"]))
    except (ImportError, OSError, ValueError, KeyError, TypeError) as exc:
        checks.append(Check("FAIL", "설정·모델 검사", str(exc)))
    for check in checks:
        print(f"[{check.status}] {check.name}: {check.detail}")
    failures = sum(check.status == "FAIL" for check in checks)
    warnings = sum(check.status == "WARN" for check in checks)
    print(f"\n점검 완료: 실패 {failures}건 / 참고 {warnings}건. 모델 추론·다운로드는 수행하지 않았습니다.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
