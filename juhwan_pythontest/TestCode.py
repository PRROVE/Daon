from re import DEBUG
from contextlib import aclosing
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

import edge_tts
import uvicorn
import whisper
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

app = FastAPI(title="음성 커뮤니티 API")
MAX_AUDIO_BYTES = 25 * 1024 * 1024
model = None
model_lock = Lock()

debug_flag = False

class TTSRequest(BaseModel):
    def __init__(self):
        super()
        if not audio:
            raise HTTPException(400, "음성 파일이 비어 있습니다.")
        if len(audio) > MAX_AUDIO_BYTES:
            raise HTTPException(413, "음성 파일은 25 MiB 이하여야 합니다.")
    model_config = {"str_strip_whitespace": True}
    text: str = Field(min_length=1, max_length=5000)

@DEBUG(debug=debug_flag)
@app.post("/audio/stt", tags=["audio"])
def speech_to_text(file: UploadFile = File(...)):
    global model
    with file.file:
        audio = file.file.read(MAX_AUDIO_BYTES + 1)
        ##
    with TemporaryDirectory() as directory:
        path = Path(directory) / "audio.wav"
        path.write_bytes(audio)
        with model_lock:
            if model is None:
                model = whisper.load_model("base")
            result = model.transcribe(str(path), fp16=(model.device.type == "cuda"))
    return {"text": result["text"].strip()}


@app.post("/audio/tts", tags=["audio"], response_class=Response)
async def text_to_speech(body: TTSRequest) -> Response:
    audio = bytearray()
    async with aclosing(edge_tts.Communicate(body.text, "ko-KR-SunHiNeural").stream()) as stream:
        async for chunk in stream:
            if chunk["type"] == "audio":
                audio.extend(chunk["data"])
    if not audio:
        raise HTTPException(502, "TTS 오디오가 생성되지 않았습니다.")
    return Response(
        bytes(audio),
        media_type="audio/mpeg",
        headers={"Content-Disposition": 'inline; filename="speech.mp3"'},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
