from supertonic import TTS
import whisper

def load_stt():
    model = whisper.load_model("small", device="cpu")
    return model

def load_tts():
    model = TTS(auto_download=True)
    voice = model.get_voice_style(voice_name="M2")
    return model, voice

def speech_to_text(model, audio_path: str) -> str:
    result = model.transcribe(audio_path, language="ko", fp16=False)
    return result["text"].strip()

def text_to_speech(model, voice, text: str):
    audio, duration = model.synthesize(text=str(text), voice_style=voice, lang="ko")
    return audio, duration

if __name__ == "__main__":
    input_file = "test1.mp4"  
    output_file = "output.wav" 

    stt_model = load_stt()
    text = speech_to_text(stt_model, input_file)
    # print("STT 결과:", text)

    tts_model, voice_style = load_tts()
    audio, duration = text_to_speech(tts_model, voice_style, text)
    tts_model.save_audio(audio, output_file)
    # print("TTS 완료:", output_file)
