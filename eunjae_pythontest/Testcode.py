from faster_whisper import WhisperModel
from supertonic import TTS


# =========================
# 1. STT 모델 준비
# =========================

stt_model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)


# =========================
# 2. Supertonic TTS 준비
# =========================

tts_model = TTS(
    auto_download=True
)

voice_style = tts_model.get_voice_style(
    voice_name="M1"
)


# =========================
# 3. STT 함수
# =========================

def speech_to_text(audio_path: str) -> str:

    segments, _ = stt_model.transcribe(
        audio_path,
        language="ko"
    )

    text = "".join(
        segment.text
        for segment in segments
    )

    return text.strip()


# =========================
# 4. TTS 함수
# =========================

def text_to_speech(
    text: str,
    output_path: str
) -> str:

    wav, duration = tts_model.synthesize(
        text=text,
        voice_style=voice_style,
        total_steps=8,
        speed=1.0,
        max_chunk_length=120,
        silence_duration=0.3,
        lang="ko",
        verbose=False
    )

    tts_model.save_audio(
        wav,
        output_path
    )

    return output_path


# =========================
# 5. 실행
# =========================

def main():

    input_audio = "test1.m4a"

    print("1. STT 시작")

    text = speech_to_text(
        input_audio
    )

    print("STT 결과:")
    print(text)

    print()

    print("2. Supertonic TTS 시작")

    output_audio = text_to_speech(
        text,
        "output.wav"
    )

    print("TTS 완료:")
    print(output_audio)


if __name__ == "__main__":
    main()