import sys
import os
sys.path.insert(0, '/cosyvoice')

from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse
from contextlib import asynccontextmanager
import torch
import torchaudio
import io

_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    from cosyvoice.cli.cosyvoice import CosyVoice2
    print("Loading CosyVoice2 model...")
    _model = CosyVoice2('/models/CosyVoice2-0.5B', load_jit=False, load_trt=False)
    print("CosyVoice2 loaded.")
    yield

app = FastAPI(lifespan=lifespan)

VOICES = ["英文女", "英文男", "中文女", "中文男", "日语男", "粤语女", "韩语女"]

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/voices")
async def voices():
    return {"voices": VOICES}

@app.post("/tts")
async def tts(body: dict):
    text = body.get("text", "")
    voice = body.get("voice", "英文女")

    if not text:
        return JSONResponse(status_code=400, content={"error": "text is required"})

    try:
        output = _model.inference_sft(text, voice, stream=False, speed=1.0)
        audio = next(output)
        wav = audio["tts_speech"]

        buf = io.BytesIO()
        torchaudio.save(buf, wav, _model.sample_rate, format="wav")
        buf.seek(0)
        return Response(content=buf.read(), media_type="audio/wav")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
