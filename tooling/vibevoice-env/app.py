import io
import os
import struct
import wave

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse

from vibevoice import VibeVoiceRealtime

SAMPLE_RATE = 24000
VOICES_DIR  = os.path.dirname(os.path.abspath(__file__))
DEVICE      = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="VibeVoice API", version="1.0.0")

print(f"--- HARDWARE: {DEVICE.upper()} ---")
print("--- LOADING VIBEVOICE-REALTIME-0.5B ---")
try:
    engine = VibeVoiceRealtime.from_pretrained("microsoft/VibeVoice-Realtime-0.5B")
    if hasattr(engine, "to"):
        engine = engine.to(DEVICE)
    print("--- ENGINE LOADED ---")
except Exception as e:
    print(f"ERROR LOADING ENGINE: {e}")
    engine = None


def _collect_pcm(text: str, preset_path: str) -> np.ndarray:
    """Collect all streaming chunks into a single int16 numpy array."""
    chunks = []
    for chunk in engine.stream(text, voice=preset_path):
        if torch.is_tensor(chunk):
            chunk = chunk.cpu().numpy()
        else:
            chunk = np.array(chunk)
        if chunk.dtype in (np.float32, np.float64):
            chunk = (chunk * 32767).astype(np.int16)
        else:
            chunk = chunk.astype(np.int16)
        chunks.append(chunk)
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.int16)


def _to_wav(pcm: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # int16 = 2 bytes
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _resolve_voice(voice: str) -> str:
    """Return absolute path to .pt file, searching current dir."""
    # Allow bare name like "fr-Spk0_man" or full filename
    name = voice if voice.endswith(".pt") else f"{voice}.pt"
    path = os.path.join(VOICES_DIR, name)
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"Voice preset not found: {path}")


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/voices")
def list_voices() -> list[str]:
    """Returns available voice names (without .pt extension)."""
    pts = [f[:-3] for f in os.listdir(VOICES_DIR) if f.endswith(".pt")]
    return sorted(pts)


@app.post("/tts")
def tts(payload: dict) -> Response:
    """POST {text, voice} → WAV audio/wav"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")
    text    = payload.get("text", "Bonjour")
    voice   = payload.get("voice", "fr-Spk1_woman")
    try:
        preset = _resolve_voice(voice)
    except FileNotFoundError as e:
        raise HTTPException(status_code=422, detail=str(e))
    pcm = _collect_pcm(text, preset)
    return Response(content=_to_wav(pcm), media_type="audio/wav")


@app.post("/v1/tts/stream")
async def tts_stream(payload: dict) -> StreamingResponse:
    """Original streaming endpoint — kept for compatibility."""
    if engine is None:
        return {"error": "Engine not loaded"}
    text     = payload.get("text", "Bonjour")
    voice    = payload.get("speakerId", "fr-Spk1_woman")
    try:
        preset = _resolve_voice(voice)
    except FileNotFoundError:
        raise HTTPException(status_code=422, detail=f"Voice not found: {voice}")

    def _gen():
        for chunk in engine.stream(text, voice=preset):
            if torch.is_tensor(chunk):
                chunk = chunk.cpu().numpy()
            else:
                chunk = np.array(chunk)
            if chunk.dtype in (np.float32, np.float64):
                chunk = (chunk * 32767).astype(np.int16)
            else:
                chunk = chunk.astype(np.int16)
            yield chunk.tobytes()

    return StreamingResponse(_gen(), media_type="audio/l16")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)