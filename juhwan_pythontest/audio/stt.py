import whisper

class STT:
    def __init__(self,model_name: str,language: str):
        self.model = whisper.load_model(model_name)
        self.language = language

    def transcribe(self,audio_path: str) -> str:
        result = self.model.transcribe(audio_path, language= self.language)
        #print(result["language"])
        return result["text"]

if __name__ == "__main__":
    stt = STT(model_name="medium",language="ko")
    print(stt.transcribe(audio_path="sample.m4a"))


