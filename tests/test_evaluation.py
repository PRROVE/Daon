import hashlib
import json
import tempfile
import unittest
import wave
from pathlib import Path

from eval.format_check import FormatChecker
from eval.report import Report


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.wav = self.folder / "sample.wav"
        self.write_wav(self.wav, b'\x01\x00' * 160)
        self.checker = FormatChecker(308, "오늘의 사연 주인공은")
        self.report = Report(self.checker, 120, {})

    @staticmethod
    def write_wav(path, samples):
        with wave.open(str(path), "wb") as file:
            file.setnchannels(1)
            file.setsampwidth(2)
            file.setframerate(16000)
            file.writeframes(samples)

    def pair(self, seconds=120, preservation=1.0):
        record = {"request_id": "example_r1", "nickname": "바다 님", "ref_transcript": "공원을 걸었어요.",
                  "story": "오늘의 사연 주인공은 바다 님이에요. 공원을 걸었어요.", "out_wav": str(self.wav),
                  "ok": True, "e2e_sec": seconds}
        state = {"nickname": record["nickname"], "reference": record["ref_transcript"], "story": record["story"]}
        judgment = {"schema": "laya-v1", "error": None, "generated": True,
                    "input_sha256": hashlib.sha256(json.dumps(state, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                    "content": {"preservation_ratio": preservation, "distortion": {"found": False}, "addition": {"found": False}},
                    "format": {key: {"pass": True} for key in ("R1", "R3", "R4", "R5", "R7")}}
        return record, judgment

    def test_length_boundary(self):
        self.assertTrue(self.checker.check("가" * 308, None)["R2"]["pass"])
        self.assertFalse(self.checker.check("가" * 309, None)["R2"]["pass"])

    def test_privacy_pattern_overrides_model_pass(self):
        record, judgment = self.pair()
        record["story"] += " 연락처는 010-1234-5678이에요."
        state = {"nickname": record["nickname"], "reference": record["ref_transcript"], "story": record["story"]}
        judgment["input_sha256"] = hashlib.sha256(json.dumps(state, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        result = self.report._merge(record, judgment)
        self.assertFalse(result["rules"]["R5"])
        self.assertFalse(result["overall_pass"])

    def test_silent_or_broken_audio_is_invalid(self):
        self.write_wav(self.wav, b'\x00\x00' * 160)
        self.assertFalse(self.checker.check("사연", self.wav)["audio"]["valid"])
        self.wav.write_bytes(b"not a WAV")
        self.assertFalse(self.checker.check("사연", self.wav)["audio"]["valid"])

    def test_e2e_boundary(self):
        self.assertTrue(self.report._merge(*self.pair(seconds=120))["overall_pass"])
        result = self.report._merge(*self.pair(seconds=120.01))
        self.assertFalse(result["e2e_success"])
        self.assertFalse(result["overall_pass"])

    def test_integrated_pass_requires_full_preservation(self):
        result = self.report._merge(*self.pair(preservation=0.9999))
        self.assertTrue(result["format_pass"])
        self.assertFalse(result["overall_pass"])

    def test_stale_judgment_is_unassessed(self):
        record, judgment = self.pair()
        record["ref_transcript"] = "전사문이 변경됐어요."
        result = self.report._merge(record, judgment)
        self.assertFalse(result["judged"])
        self.assertIsNone(result["fact_recall"])
        self.assertTrue(self.report._content([result])["provisional"])

    def test_empty_denominator_is_not_zero_percent(self):
        result = self.report._content([])
        self.assertIsNone(result["fact_recall_macro"]["value"])
        self.assertIsNone(result["distortion_rate"]["value"])


if __name__ == "__main__":
    unittest.main()
