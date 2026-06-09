import io
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import numpy as np

app = FastAPI()

MODEL_NAME = "parler-tts/parler-tts-mini-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Voice descriptions for the model
VOICE_DESCRIPTIONS = {
    "female": (
        "Une voix féminine française, calme et naturelle, avec une diction claire "
        "et une prosodie légèrement expressive. Enregistrement de qualité studio, "
        "sans bruit de fond."
    ),
    "male": (
        "Une voix masculine française, profonde et posée, avec une articulation nette "
        "et un débit modéré. Enregistrement de qualité studio, sans bruit de fond."
    ),
    "female_expressive": (
        "Une voix féminine française très expressive et animée, avec des variations "
        "d'intonation marquées. Enregistrement de haute qualité."
    ),
    "male_deep": (
        "Une voix masculine française grave et autoritaire, avec une diction précise "
        "et un débit lent. Enregistrement de qualité studio."
    ),
}

_model = None
_tokenizer = None


def load_model():
    global _model, _tokenizer
    print(f"Loading Parler-TTS model '{MODEL_NAME}' on {DEVICE}...")
    _model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Model loaded.")


@app.on_event("startup")
def startup():
    load_model()


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model": MODEL_NAME}


@app.get("/voices")
def list_voices():
    return {"voices": list(VOICE_DESCRIPTIONS.keys())}


class TTSRequest(BaseModel):
    text: str
    voice: str = "female"
    description: str | None = None


@app.post("/tts")
def synthesize(req: TTSRequest):
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Use custom description or lookup by voice name
    if req.description:
        description = req.description
    elif req.voice in VOICE_DESCRIPTIONS:
        description = VOICE_DESCRIPTIONS[req.voice]
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{req.voice}' not found. Available: {list(VOICE_DESCRIPTIONS.keys())}"
        )

    input_ids = _tokenizer(description, return_tensors="pt").input_ids.to(DEVICE)
    prompt_ids = _tokenizer(req.text, return_tensors="pt").input_ids.to(DEVICE)

    with torch.no_grad():
        generation = _model.generate(
            input_ids=input_ids,
            prompt_input_ids=prompt_ids,
        )

    audio = generation.cpu().numpy().squeeze()
    sample_rate = _model.config.sampling_rate

    bio = io.BytesIO()
    sf.write(bio, audio, sample_rate, format="WAV")
    bio.seek(0)
    return Response(content=bio.read(), media_type="audio/wav")
