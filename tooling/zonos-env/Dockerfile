# Zonos TTS — CPU image
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# espeak-ng in separate layer avoids stale-cache issues
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng espeak-ng-data \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Zonos requires torch>=2.5.1 — install latest stable CPU build
RUN pip install --no-cache-dir torch==2.5.1+cpu torchaudio==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Zonos from git — patch pyproject.toml so subpackages (zonos.backbone etc.) are included
RUN git clone --depth 1 https://github.com/Zyphra/Zonos.git /tmp/zonos \
    && sed -i 's/include = \["zonos"\]/include = ["zonos", "zonos.*"]/' /tmp/zonos/pyproject.toml \
    && pip install --no-cache-dir /tmp/zonos \
    && rm -rf /tmp/zonos

COPY app.py .

# /app/voices: mount WAV reference files here for voice cloning
VOLUME ["/app/voices"]

ENV DEVICE=cpu
EXPOSE 8010

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8010"]
