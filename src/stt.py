from pathlib import Path

import numpy as np
import torch
import whisper

from config.paths import DATA_DIR
from config.settings import Settings

SAMPLE_RATE = 16000


class STT:
    """Silero VAD 로 쉼에서 나눈 조각을 openai-whisper 로 전사해 한국어 전사문을 만든다."""

    def __init__(self, model: str, language: str, model_dir: Path, device: str, decode_options: dict,
                 vad_dir: Path, max_chunk_sec: float, long_pause_sec: float) -> None:
        self.model_name = model
        self.language = language
        self.decode_options = decode_options
        self.max_chunk_sec = max_chunk_sec
        self.long_pause_sec = long_pause_sec
        # openai-whisper 이름(turbo 등)이 아니면 model_dir 의 변환 가중치(komixv2.pt) 파일
        name = model if model in whisper.available_models() else str(model_dir / f"{model}.pt")
        # device "auto" 는 whisper 기본 동작(cuda 있으면 cuda)에 맡긴다
        self.model = whisper.load_model(name, device=None if device == "auto" else device, download_root=str(model_dir))
        # Jetson 에서 인터넷 없이 돌도록 GitHub 대신 models/vad 에 받아 둔 저장소에서 읽는다
        self.vad, utils = torch.hub.load(repo_or_dir=str(vad_dir), model="silero_vad", source="local")
        self.get_speech_timestamps = utils[0]

    def transcribe(self, audio_path: Path) -> str:
        """audio_path를 전사해 문자열로 돌려준다."""
        audio = whisper.load_audio(str(audio_path))
        bounds = [0.0, *self._cut_points(audio), len(audio) / SAMPLE_RATE]
        texts = [self.model.transcribe(audio[int(a * SAMPLE_RATE):int(b * SAMPLE_RATE)], language=self.language,
                                       **self.decode_options)["text"].strip() for a, b in zip(bounds, bounds[1:])]
        return " ".join(texts)

    def _cut_points(self, audio: np.ndarray) -> list[float]:
        """자를 시각 목록. 소리는 버리지 않고 VAD 가 찾은 쉼의 가운데에서만 나눈다."""
        total = len(audio) / SAMPLE_RATE
        if total <= self.max_chunk_sec:
            return []
        ts = self.get_speech_timestamps(torch.from_numpy(audio), self.vad, sampling_rate=SAMPLE_RATE, return_seconds=True)
        gaps = [((a["end"] + b["start"]) / 2, b["start"] - a["end"]) for a, b in zip(ts, ts[1:])]
        cuts, start = [], 0.0
        while total - start > self.max_chunk_sec:
            # 너무 짧은 조각을 피하려고 한도의 절반 이후 쉼만 후보로 둔다
            cand = [g for g in gaps if start + self.max_chunk_sec / 2 <= g[0] <= start + self.max_chunk_sec]
            # 방언 라벨 대조에서 단어 안의 쉼(말 늘임)은 최대 0.8초라 그보다 긴 쉼을 먼저 고른다
            long = [g for g in cand if g[1] >= self.long_pause_sec]
            cuts.append(max(long or cand, key=lambda g: g[1])[0] if cand else start + self.max_chunk_sec)
            start = cuts[-1]
        return cuts


if __name__ == "__main__":
    s = Settings()
    stt = STT(device=s.device, **s.stt)
    print(stt.transcribe(DATA_DIR / "eval_set" / "s44.wav"))
