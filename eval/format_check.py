import re
from pathlib import Path

import numpy as np
import soundfile as sf

from config.paths import RESULTS_DIR
from config.settings import Settings

# R5 코드 검사: 전화번호·주민등록번호·집 주소·병원명. 실명은 정규식으로 못 잡아 LLM 판정과 병행한다
R5_PATTERNS = {
    "phone": re.compile(r"0\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4}"),
    "resident_id": re.compile(r"\d{6}[-\s]?\d{7}"),
    "address": re.compile(r"\d+\s*번지|\d+\s*동\s*\d+\s*호|[가-힣]+(?:로|길)\s*\d+(?:번길|번지|호|-\d+)"),
    "hospital": re.compile(r"[가-힣A-Za-z]+(?:병원|의원|클리닉)"),
}
# 특정 기관을 가리키지 않는 일반 명사. "지정병원" 같은 말이 병원명으로 오탐되지 않게 뺀다
GENERIC_HOSPITAL = {"지정병원", "종합병원", "대학병원", "동네병원", "요양병원", "개인병원", "큰병원"}


class FormatChecker:
    """평가지표 ②의 코드 검사: R2 글자 수 상한, R5 정규식, R6 도입 문구, ③의 음성 정상 생성 여부."""

    def __init__(self, max_chars: int, intro: str) -> None:
        self.max_chars = max_chars
        self.intro = intro

    def check(self, story: str | None, wav_path: str | Path | None, reference: str = "") -> dict:
        """후처리된 사연 텍스트와 출력 wav 경로를 받아 규칙별 결과를 돌려준다."""
        text = story or ""
        found = {k: [h for h in p.findall(text) if h not in GENERIC_HOSPITAL] for k, p in R5_PATTERNS.items()}
        hits = {k: v for k, v in found.items() if v}
        # 원문에서 확인된 기관명이 사연에서 띄어 쓰인 경우만 추가 대조한다.
        names = set(R5_PATTERNS["hospital"].findall(reference)) - GENERIC_HOSPITAL
        spaced = sorted(name for name in names if name in re.sub(r"\s+", "", text))
        if spaced:
            hits["hospital_reference"] = spaced
        audio = self._audio(wav_path)
        return {
            "R2": {"pass": len(text) <= self.max_chars, "chars": len(text)},
            "R5_regex": {"pass": not hits, "hits": hits},
            "R6": {"pass": text.startswith(self.intro)},
            "audio": audio,
        }

    @staticmethod
    def _audio(wav_path: str | Path | None) -> dict:
        # 평가지표 ③ "재생 가능한 음성": 파일 존재·비어 있지 않음·디코딩 가능·전체 무음 아님
        if not wav_path or not Path(wav_path).exists() or Path(wav_path).stat().st_size == 0:
            return {"valid": False, "duration": None, "reason": "파일 없음 또는 빈 파일"}
        try:
            data, sr = sf.read(wav_path)
        except RuntimeError as e:
            return {"valid": False, "duration": None, "reason": f"디코딩 불가: {e}"}
        duration = len(data) / sr
        if not np.any(data):
            return {"valid": False, "duration": duration, "reason": "전체 무음"}
        return {"valid": True, "duration": duration, "reason": None}


if __name__ == "__main__":
    s = Settings()
    checker = FormatChecker(s.llm["target_chars"]["max"], s.eval["intro"])
    story = "오늘의 사연 주인공은 바다 님이에요. 서울병원에 다녀왔어요. 연락처는 010-1234-5678이에요."
    print(checker.check(story, RESULTS_DIR / "_smoke" / "story.wav"))
