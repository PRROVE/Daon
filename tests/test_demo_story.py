import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import demo_story as demo


class DemoInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name).resolve()
        self.audio = self.folder / "sample.wav"
        self.audio.write_bytes(b"input validation fixture")

    def test_own_audio_needs_no_manifest(self):
        args = demo.make_parser().parse_args(["--audio", str(self.audio), "--nickname", "바다 님"])
        item = demo.resolve_input(args)
        self.assertEqual(item.audio_path, self.audio)
        self.assertEqual(item.nickname, "바다 님")
        self.assertIsNone(item.sample_id)

    def test_manifest_paths_are_relative_to_manifest(self):
        manifest = self.folder / "manifest.jsonl"
        manifest.write_text('\n' + json.dumps({"sample_id": "c04", "wav": "sample.wav", "nickname": "이슬 님"}) + '\n', encoding="utf-8-sig")
        item = demo.resolve_input(demo.make_parser().parse_args(["--sample", "c04", "--manifest", str(manifest)]))
        self.assertEqual((item.audio_path, item.nickname, item.sample_id), (self.audio, "이슬 님", "c04"))

    def test_missing_default_dataset_explains_audio_option(self):
        with patch.object(demo, "ROOT", self.folder), self.assertRaisesRegex(ValueError, "--audio"):
            demo.resolve_input(demo.make_parser().parse_args([]))

    def test_duplicate_sample_is_rejected(self):
        manifest = self.folder / "manifest.jsonl"
        row = json.dumps({"sample_id": "c04", "wav": "sample.wav", "nickname": "바다 님"})
        manifest.write_text(row + '\n' + row, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "정확히 한 번"):
            demo.resolve_input(demo.make_parser().parse_args(["--manifest", str(manifest)]))

    def test_invalid_json_has_line_number(self):
        manifest = self.folder / "manifest.jsonl"
        manifest.write_text('\n{broken}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "2행"):
            demo.resolve_input(demo.make_parser().parse_args(["--manifest", str(manifest)]))

    def test_empty_nickname_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "nickname"):
            demo.resolve_input(demo.make_parser().parse_args(["--audio", str(self.audio), "--nickname", " "]))

    def test_audio_and_sample_are_mutually_exclusive(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            demo.make_parser().parse_args(["--audio", str(self.audio), "--sample", "c04"])
        self.assertEqual(raised.exception.code, 2)

    def test_help_and_missing_file_work_without_model_packages(self):
        script = str(demo.ROOT / "scripts/demo_story.py")
        for args, code in [(["--help"], 0), (["--audio", str(self.folder / "missing.wav")], 2)]:
            with self.subTest(args=args):
                result = subprocess.run([sys.executable, "-S", script, *args], capture_output=True, timeout=10)
                self.assertEqual(result.returncode, code, result.stderr.decode(errors="replace"))
                self.assertNotIn(b"ModuleNotFoundError", result.stderr)

    def test_external_output_does_not_turn_completed_tts_into_failure(self):
        class FakePipeline:
            def __init__(self, settings):
                pass

            def _timed(self, record, stage, key, fn):
                record[key], record[stage + "_sec"] = fn(), 0.01
                return True

        class ReportingPipeline(demo.DemoProgress, FakePipeline):
            pass

        record = {}
        output = self.folder / "audio/story.wav"
        pipeline = ReportingPipeline(None, self.folder)
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertTrue(pipeline._timed(record, "tts", "out_wav", lambda: str(output)))
        self.assertIn(str(output), stdout.getvalue())
        events = [json.loads(line) for line in (self.folder / "events.jsonl").read_text().splitlines()]
        self.assertEqual([item["event"] for item in events], ["stage_start", "stage_end"])
        self.assertTrue(events[-1]["ok"])


if __name__ == "__main__":
    unittest.main()
