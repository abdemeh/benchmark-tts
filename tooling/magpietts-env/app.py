import io

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI()

SPEAKERS = {
    "Sofia": 1,   # English female — works well for French
    "Aria": 2,    # English female
    "Jason": 3,   # English male
    "Leo": 4,     # English male
    "John": 0,    # English male (public domain LibriVox reader)
}
VOICES = list(SPEAKERS.keys())
SAMPLE_RATE = 22050  # NanoCodec output rate

_model = None


@app.on_event("startup")
def load():
    global _model
    from nemo.collections.tts.models import MagpieTTSModel
    _model = MagpieTTSModel.from_pretrained("nvidia/magpie_tts_multilingual_357m")
    _model = _model.cuda()
    _model.eval()
    print("MagpieTTS loaded successfully")


@app.get("/health")
def health():
    return {"status": "ok", "device": "cuda" if torch.cuda.is_available() else "cpu"}


@app.get("/voices")
def voices():
    return {"voices": VOICES}


class TTSRequest(BaseModel):
    text: str
    voice: str = "Sofia"
    language: str = "fr"


@app.post("/tts")
def synthesize(req: TTSRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    speaker_idx = SPEAKERS.get(req.voice, 1)
    lang = req.language if req.language in ("fr", "en", "es", "de", "it", "zh", "hi", "ja", "vi") else "fr"

    with torch.no_grad():
        audio, audio_len = _model.do_tts(
            req.text,
            language=lang,
            apply_TN=True,  # built-in text normalization (handles numbers, abbrevs)
            speaker_index=speaker_idx,
        )

    # audio shape may be [T] or [1, T]
    audio_np = audio.squeeze()[:audio_len].cpu().float().numpy()

    buf = io.BytesIO()
    sf.write(buf, audio_np, SAMPLE_RATE, format="WAV")
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")
