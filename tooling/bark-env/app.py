import io
import numpy as np
import torch
import scipy.io.wavfile
from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse
from contextlib import asynccontextmanager

_processor = None
_model = None

# French voice presets (Bark speaker embeddings)
VOICES = {
    "fr_speaker_1": "v2/fr_speaker_1",
    "fr_speaker_3": "v2/fr_speaker_3",
    "fr_speaker_6": "v2/fr_speaker_6",
    "fr_speaker_9": "v2/fr_speaker_9",
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _processor, _model
    from transformers import AutoProcessor, AutoModel
    print("Loading Bark model...")
    _processor = AutoProcessor.from_pretrained("suno/bark-small")
    _model = AutoModel.from_pretrained("suno/bark-small")
    _model = _model.to("cuda" if torch.cuda.is_available() else "cpu")
    print("Bark loaded.")
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/voices")
def voices():
    return {"voices": list(VOICES.keys())}

@app.post("/tts")
async def tts(body: dict):
    text = body.get("text", "")
    voice = body.get("voice", "fr_speaker_3")

    if not text:
        return JSONResponse(status_code=400, content={"error": "text required"})

    preset = VOICES.get(voice, "v2/fr_speaker_3")

    try:
        inputs = _processor(text, voice_preset=preset, return_tensors="pt")
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            speech_values = _model.generate(**inputs, do_sample=True)

        sr = _model.generation_config.sample_rate
        audio = speech_values.cpu().numpy().squeeze()
        audio_int16 = (audio * 32767).astype(np.int16)

        buf = io.BytesIO()
        scipy.io.wavfile.write(buf, sr, audio_int16)
        buf.seek(0)
        return Response(content=buf.read(), media_type="audio/wav")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
