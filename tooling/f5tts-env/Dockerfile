# F5-TTS — CPU image
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# TorchCodec is the default torchaudio 2.5 audio backend — install CPU build
RUN pip install --no-cache-dir torchcodec \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY app.py .

# /app/voices: place <name>.wav + <name>.txt (transcript) pairs here
# Example: french-female.wav + french-female.txt
VOLUME ["/app/voices"]

ENV DEVICE=cpu
# F5TTS_MODEL: change to a French fine-tune if available
# e.g. RASPIAUDIO/f5tts-French or a local path
ENV F5TTS_MODEL=F5TTS_v1_Base
EXPOSE 8012

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8012"]
