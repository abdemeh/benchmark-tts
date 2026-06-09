import io
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import soundfile as sf
import numpy as np
from melo.api import TTS

app = FastAPI()

# Pre-load model at startup
DEVICE = "cuda"
_models: dict = {}

SUPPORTED_LANGS = {
    "FR": "FR",
    "EN": "EN",
    "ES": "ES",
    "ZH": "ZH",
    "JP": "JP",
    "KR": "KR",
}


def get_model(lang: str) -> tuple:
    if lang not in _models:
        model = TTS(language=lang, device=DEVICE)
        speaker_ids = model.hps.data.spk2id
        _models[lang] = (model, speaker_ids)
    return _models[lang]


@app.on_event("startup")
def startup():
    # Pre-load French model
    get_model("FR")


@app.get("/voices")
def list_voices():
    model, speaker_ids = get_model("FR")
    return {"voices": list(speaker_ids.keys())}


@app.get("/health")
def health():
    return {"status": "ok"}


class TTSRequest(BaseModel):
    text: str
    voice: str = "FR"
    speed: float = 1.0


@app.post("/tts")
def synthesize(req: TTSRequest):
    lang = req.voice if req.voice in SUPPORTED_LANGS else "FR"
    model, speaker_ids = get_model(lang)

    # Get first speaker if lang used as voice, else look up by name
    if req.voice in SUPPORTED_LANGS:
        speaker_id = list(speaker_ids.values())[0]
    elif req.voice in speaker_ids:
        speaker_id = speaker_ids[req.voice]
    else:
        raise HTTPException(status_code=400, detail=f"Voice '{req.voice}' not found")

    # Synthesize to buffer
    bio = io.BytesIO()
    model.tts_to_file(
        req.text,
        speaker_id,
        bio,
        speed=req.speed,
        format="wav",
    )
    bio.seek(0)
    return Response(content=bio.read(), media_type="audio/wav")
