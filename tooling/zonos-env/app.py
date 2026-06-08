"""
Zonos TTS FastAPI server — Orion voice benchmark
Model: Zyphra/Zonos-v0.1-transformer (Apache 2.0)
Supports: multilingual including French, emotion control, voice cloning via WAV reference
"""
from __future__ import annotations

import io
import os
import logging
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("zonos-api")

DEVICE = os.getenv("DEVICE", "cpu")
VOICES_DIR = Path("/app/voices")
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "fr-default")

app = FastAPI(title="Zonos TTS API", version="1.0.0")

model = None
speaker_embeddings: dict[str, torch.Tensor] = {}


@app.on_event("startup")
async def startup() -> None:
    global model, speaker_embeddings
    log.info("Loading Zonos model on %s ...", DEVICE)
    from zonos.model import Zonos
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=DEVICE)
    model.eval()
    log.info("Zonos model loaded.")
    _load_speaker_embeddings()


def _load_speaker_embeddings() -> None:
    """Pre-compute speaker embeddings from any WAV files found in VOICES_DIR."""
    if not VOICES_DIR.exists():
        log.warning("Voices dir %s not found — using random speaker.", VOICES_DIR)
        return
    for wav_path in VOICES_DIR.glob("*.wav"):
        try:
            wav, sr = torchaudio.load(str(wav_path))
            wav = wav.to(DEVICE)
            embedding = model.make_speaker_embedding(wav, sr)
            key = wav_path.stem
            speaker_embeddings[key] = embedding
            log.info("Loaded speaker: %s", key)
        except Exception as exc:
            log.error("Failed to load speaker %s: %s", wav_path, exc)


class TtsRequest(BaseModel):
    text: str
    voice: Optional[str] = "fr-default"
    speed: Optional[float] = 1.0
    language: Optional[str] = "fr-fr"
    emotion_strength: Optional[float] = 0.5


class VoicesResponse(BaseModel):
    voices: list[str]


@app.post("/tts")
def tts_endpoint(req: TtsRequest) -> Response:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    from zonos.conditioning import make_cond_dict

    speaker = speaker_embeddings.get(req.voice or DEFAULT_VOICE)

    cond_dict = make_cond_dict(
        text=req.text,
        language=req.language or "fr-fr",
        speaker=speaker,
        speaking_rate=max(5.0, min(30.0, (req.speed or 1.0) * 15.0)),
        device=DEVICE,
    )
    conditioning = model.prepare_conditioning(cond_dict)

    with torch.no_grad():
        codes = model.generate(conditioning)
        wavs = model.autoencoder.decode(codes).cpu()

    wav = wavs[0, 0].unsqueeze(0)  # [1, samples]
    sample_rate = model.autoencoder.sampling_rate

    buf = io.BytesIO()
    torchaudio.save(buf, wav, sample_rate, format="wav")
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")


@app.get("/voices")
def list_voices() -> list[str]:
    known = list(speaker_embeddings.keys())
    return known if known else ["fr-default"]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": model is not None, "device": DEVICE}
