import io
import sys
import wave

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

# CosyVoice submodule paths
sys.path.insert(0, "/cosyvoice")
sys.path.insert(0, "/cosyvoice/third_party/AcademiCodec")
sys.path.insert(0, "/cosyvoice/third_party/Matcha-TTS")

from cosyvoice.cli.cosyvoice import CosyVoice2

app = FastAPI()

# CosyVoice2-0.5B SFT voices (multilingual model — French text works with these)
VOICES = ["中文女", "中文男", "英文女", "英文男", "日语男", "粤语女", "韩语女"]

_model = None


@app.on_event("startup")
def load():
    global _model
    _model = CosyVoice2(
        "/models/CosyVoice2-0.5B",
        load_jit=False,
        load_trt=False,
        fp16=False,
    )
    print(f"CosyVoice2 loaded — sample_rate={_model.sample_rate}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/voices")
def voices():
    return {"voices": VOICES}


class TTSRequest(BaseModel):
    text: str
    voice: str = "英文女"


@app.post("/tts")
def synthesize(req: TTSRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    voice = req.voice if req.voice in VOICES else "英文女"

    chunks = []
    for _, result in enumerate(_model.inference_sft(req.text, voice, stream=False)):
        # result["tts_speech"] is a float32 tensor of shape [1, T], values in [-1, 1]
        chunks.append(result["tts_speech"].squeeze(0).numpy())

    if not chunks:
        raise HTTPException(status_code=500, detail="No audio generated")

    audio = np.concatenate(chunks, axis=0)
    sr = _model.sample_rate

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframesraw(
            (audio * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
        )
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")
