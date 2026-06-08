"""
F5-TTS FastAPI server — Orion voice benchmark
Model: SWivid/F5-TTS (CC BY-NC — non-commercial use only)
Supports: Zero-shot voice cloning via WAV reference. Base model = EN+ZH.
French support requires a French fine-tune (e.g. RASPIAUDIO/f5tts-French).
"""
from __future__ import annotations

import io
import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("f5tts-api")

DEVICE = os.getenv("DEVICE", "cpu")
VOICES_DIR = Path("/app/voices")
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "default")
MODEL_NAME  = os.getenv("F5TTS_MODEL", "F5TTS_Base")
CKPT_FILE   = os.getenv("F5TTS_CKPT_FILE", "")   # hf:// path or local path
VOCAB_FILE  = os.getenv("F5TTS_VOCAB_FILE", "")  # hf:// path or local path

app = FastAPI(title="F5-TTS API", version="1.0.0")

tts_model = None
# voice_name -> (ref_audio_path, ref_text)
voice_refs: dict[str, tuple[str, str]] = {}


def _resolve_hf_path(uri: str) -> str:
    """hf://owner/repo/file → local cached path via huggingface_hub."""
    if not uri or not uri.startswith("hf://"):
        return uri
    parts = uri[5:].split("/")
    repo_id  = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    from huggingface_hub import hf_hub_download
    local = hf_hub_download(repo_id=repo_id, filename=filename)
    log.info("Resolved %s → %s", uri, local)
    return local


@app.on_event("startup")
async def startup() -> None:
    global tts_model
    log.info("Loading F5-TTS model %s on %s ...", MODEL_NAME, DEVICE)
    from f5_tts.api import F5TTS
    tts_model = F5TTS(
        model=MODEL_NAME,
        ckpt_file=_resolve_hf_path(CKPT_FILE),
        vocab_file=_resolve_hf_path(VOCAB_FILE),
        device=DEVICE,
    )
    log.info("F5-TTS loaded.")
    _load_voice_refs()
    # If no custom voices mounted, scan known locations for a bundled example
    if "default" not in voice_refs:
        _BUILTIN_CANDIDATES = [
            ("/app/default_voice/default.wav", Path("/app/default_voice/default.txt")),
            *[
                (None, p) for p in [
                    "infer/examples/en/en_1_ref_short.wav",
                    "infer/examples/basic/basic_ref_en.wav",
                    "test_en_1_ref_short.wav",
                ]
            ],
        ]
        _BUILTIN_TEXT = "Some call me nature, others call me mother nature."
        # Check the baked-in path first
        _baked = Path("/app/default_voice/default.wav")
        if _baked.exists():
            _txt = Path("/app/default_voice/default.txt")
            voice_refs["default"] = (
                str(_baked),
                _txt.read_text(encoding="utf-8").strip() if _txt.exists() else _BUILTIN_TEXT,
            )
            log.info("Using baked-in reference audio: %s", _baked)
        else:
            # Use importlib.resources — the official way (works with any install)
            try:
                from importlib.resources import files as _res_files
                _ref = _res_files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav")
                _ref_path = str(_ref)
                voice_refs["default"] = (_ref_path, _BUILTIN_TEXT)
                log.info("Using package ref via importlib.resources: %s", _ref_path)
            except Exception as _e:
                log.warning("Could not load built-in reference: %s", _e)


def _load_voice_refs() -> None:
    """
    Load voice reference pairs from VOICES_DIR.
    Each voice needs: <name>.wav  and  <name>.txt (the transcript of the WAV).
    """
    if not VOICES_DIR.exists():
        log.warning("Voices dir %s not found — built-in reference only.", VOICES_DIR)
        return
    for wav_path in VOICES_DIR.glob("*.wav"):
        txt_path = wav_path.with_suffix(".txt")
        ref_text = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""
        voice_refs[wav_path.stem] = (str(wav_path), ref_text)
        log.info("Loaded voice ref: %s", wav_path.stem)


class TtsRequest(BaseModel):
    text: str
    voice: Optional[str] = "default"
    speed: Optional[float] = 1.0
    nfe_step: Optional[int] = 32   # diffusion steps: 16=fast, 32=quality


@app.post("/tts")
def tts_endpoint(req: TtsRequest) -> Response:
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    voice_key = req.voice or DEFAULT_VOICE
    ref_audio, ref_text = voice_refs.get(voice_key) or voice_refs.get("default", (None, None))
    if ref_audio is None:
        raise HTTPException(
            status_code=422,
            detail="No voice reference loaded. Mount <name>.wav + <name>.txt pairs at /app/voices.",
        )

    wav, sr, _ = tts_model.infer(
        ref_file=ref_audio,
        ref_text=ref_text,
        gen_text=req.text,
        speed=req.speed or 1.0,
        nfe_step=req.nfe_step or 32,
        cross_fade_duration=0.15,
    )

    buf = io.BytesIO()
    sf.write(buf, np.array(wav), sr, format="WAV")
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")


@app.get("/voices")
def list_voices() -> list[str]:
    return list(voice_refs.keys())


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": tts_model is not None, "device": DEVICE}
