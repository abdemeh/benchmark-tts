import io
import torch
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse
from contextlib import asynccontextmanager

_model = None
MODEL_PATH = "/models/Qwen3-TTS-0.6B"

SPEAKERS = ["Ryan", "Aiden", "Serena", "Vivian"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    from qwen_tts import Qwen3TTSModel
    print("Loading Qwen3-TTS 0.6B CustomVoice...")
    _model = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map="cuda:0",
        dtype=torch.float16,  # T4 is Turing arch — no native bfloat16, use float16
    )
    print("Qwen3-TTS loaded.")
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/voices")
def voices():
    return {"voices": SPEAKERS}

@app.post("/tts")
async def tts(body: dict):
    text = body.get("text", "")
    voice = body.get("voice", "Ryan")
    language = body.get("language", "French")

    if not text:
        return JSONResponse(status_code=400, content={"error": "text required"})

    try:
        with torch.no_grad():
            wavs, sr = _model.generate_custom_voice(
                text=text,
                language=language,
                speaker=voice,
            )
        buf = io.BytesIO()
        sf.write(buf, wavs[0], sr, format="WAV")
        buf.seek(0)
        return Response(content=buf.read(), media_type="audio/wav")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
