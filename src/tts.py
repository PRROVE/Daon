from pathlib import Path

import supertonic

from config.paths import RESULTS_DIR
from config.settings import Settings


class TTS:
    """Supertonic으로 텍스트를 wav 파일로 합성한다."""

    def __init__(self, voice: str, lang: str, model_dir: Path) -> None:
        self.lang = lang
        self.tts = supertonic.TTS(model_dir=str(model_dir), auto_download=True)
        self.style = self.tts.get_voice_style(voice_name=voice)

    def synthesize(self, text: str, out_path: Path) -> Path:
        """text를 합성해 out_path에 저장하고 그 경로를 돌려준다."""
        wav, _ = self.tts.synthesize(text=text, voice_style=self.style, lang=self.lang)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.tts.save_audio(wav, str(out_path))
        return out_path


if __name__ == "__main__":
    cfg = Settings().tts
    tts = TTS(voice=cfg["voice"], lang=cfg["lang"], model_dir=cfg["model_dir"])
    out = tts.synthesize(
        "오늘의 사연 주인공은 바다 님이에요. 아침마다 공원을 걸으며 하루를 시작해요.",
        RESULTS_DIR / "_smoke" / "tts.wav",
    )
    print(out, out.stat().st_size, "bytes")
