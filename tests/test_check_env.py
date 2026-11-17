import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import URLError

from scripts import check_env


class EnvironmentChecksTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def test_missing_empty_and_changed_files_fail(self):
        path = self.folder / "prompt.md"
        self.assertEqual(check_env.check_file("prompt", path).status, "FAIL")
        path.touch()
        self.assertEqual(check_env.check_file("prompt", path).status, "FAIL")
        path.write_bytes(b"original\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(check_env.check_file("prompt", path, digest).status, "OK")
        path.write_bytes(b"original\r\n")
        self.assertEqual(check_env.check_file("prompt", path, digest).status, "FAIL")

    def test_cuda_torch_build_matches_pinned_base_version(self):
        (self.folder / "requirements.txt").write_text("torch==2.11.0\nlaya==0.4.22\n", encoding="utf-8")
        with patch.object(check_env, "ROOT", self.folder), patch.object(check_env, "version", return_value="2.11.0+cu128"):
            checks = check_env.check_packages(generation_only=True)
        self.assertEqual([(item.name, item.status) for item in checks], [("torch", "OK")])

    def test_generation_only_does_not_require_laya(self):
        for name in ("story.md", "whisper/komixv2.pt", "vad/hubconf.py", "vad/src/silero_vad/data/silero_vad.jit"):
            path = self.folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        settings = SimpleNamespace(
            stt={"model_dir": self.folder / "whisper", "model": "komixv2", "vad_dir": self.folder / "vad"},
            llm={"prompt_path": self.folder / "story.md", "prompt_sha256": hashlib.sha256(b"fixture").hexdigest()},
            tts={"model_dir": self.folder / "tts", "voice": "M1"},
        )
        checks = check_env.check_models(settings, generation_only=True)
        self.assertFalse(any(item.status == "FAIL" for item in checks))
        self.assertEqual(checks[-1].status, "WARN")  # TTS can download on its first run.

    def test_registered_ollama_model_passes(self):
        with patch.object(check_env, "urlopen", return_value=io.BytesIO(json.dumps({"models": [{"name": "qwen3:4b-instruct"}]}).encode())) as request:
            result = check_env.check_ollama("http://localhost:11434/", "qwen3:4b-instruct")
        self.assertEqual(result.status, "OK")
        request.assert_called_once_with("http://localhost:11434/api/tags", timeout=3)

    def test_missing_ollama_model_fails(self):
        with patch.object(check_env, "urlopen", return_value=io.BytesIO(b'{"models": []}')):
            self.assertEqual(check_env.check_ollama("http://localhost:11434", "qwen3:4b-instruct").status, "FAIL")

    def test_stopped_ollama_has_actionable_failure(self):
        with patch.object(check_env, "urlopen", side_effect=URLError("connection refused")):
            result = check_env.check_ollama("http://localhost:11434", "qwen3:4b-instruct")
        self.assertEqual(result.status, "FAIL")
        self.assertIn("ollama serve", result.detail)

    def test_invalid_ollama_response_fails(self):
        with patch.object(check_env, "urlopen", return_value=io.BytesIO(b'[]')):
            self.assertEqual(check_env.check_ollama("http://localhost:11434", "qwen3:4b-instruct").status, "FAIL")


if __name__ == "__main__":
    unittest.main()
