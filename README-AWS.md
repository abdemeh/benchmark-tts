# Orion Benchmark TTS/STT — AWS Deployment Guide

## Target machine

**AWS g4dn.xlarge** — 4 vCPU, 16 GB RAM, 1× NVIDIA T4 (16 GB VRAM), Ubuntu 22.04

---

## Architecture

```
benchmark-tts/
├── setup-aws.sh                         ← run this first on the instance
├── scripts/orion-benchmark/
│   ├── benchmark-aws.py                 ← interactive benchmark runner
│   ├── requirements.txt                 ← requests, colorama
│   ├── results/                         ← JSON output (feed to orion-benchmark web)
│   └── wavs/<engine>/                   ← generated audio files
└── tooling/
    ├── piper-env/        CPU   port 5000
    ├── kokoro-env/       CPU   port 8880
    ├── whisper-env/      CPU   port 8001
    ├── coqui-env/        GPU   port 8000
    ├── chatterbox-env/   GPU   port 8009
    ├── zonos-env/        GPU   port 8010
    ├── f5tts-env/        GPU   port 8012
    └── vibevoice-env/    GPU   port 8020
```

Each engine has its own `docker-compose.yml` (CPU) and/or `docker-compose.gpu.yml` (GPU).  
`benchmark-aws.py` starts one container at a time, waits for it to be healthy, runs all phrases × voices × runs, then stops it before moving to the next.

---

## Step 1 — Connect to the instance

```bash
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>
```

Make sure port **22** is open in the security group inbound rules.

---

## Step 2 — Clone and run setup

```bash
# Clone
git clone https://github.com/abdemeh/benchmark-tts.git ~/benchmark-tts
cd ~/benchmark-tts

# Run setup (installs Docker, NVIDIA Container Toolkit, Python venv)
chmod +x setup-aws.sh
./setup-aws.sh
```

> If you see a message about the `docker` group, run `newgrp docker` or log out/in before continuing.

---

## Step 3 — Run the benchmark

```bash
source ~/benchmark-env/bin/activate
cd ~/benchmark-tts/scripts/orion-benchmark
python3 benchmark-aws.py
```

The script will:
1. Detect the T4 GPU automatically
2. Show a numbered menu to select which engines to test
3. Ask how many runs per sentence (default 3)
4. Ask whether to create a new result file or append to an existing one
5. Start each Docker container, wait for it to be ready (up to 10 min for GPU model downloads), run all measurements, then stop the container
6. Save incrementally — safe to Ctrl+C and resume

---

## Step 4 — Retrieve results

Option A — copy the JSON back to your machine:

```bash
# From your local machine
scp -i your-key.pem ubuntu@<EC2_PUBLIC_IP>:~/benchmark-tts/scripts/orion-benchmark/results/*.json .
```

Option B — print it on the server:

```bash
cat ~/benchmark-tts/scripts/orion-benchmark/results/<your-file>.json
```

Then drop the JSON into the **orion-benchmark** web app (`web/orion-benchmark`) to visualise the results.

---

## Engines overview

| Engine | Type | HW | Port | Notes |
|---|---|---|---|---|
| Piper | TTS | CPU | 5000 | Pre-built image, fast |
| Kokoro | TTS | CPU | 8880 | Pre-built image, high quality |
| Whisper | STT | CPU | 8001 | Pre-built image, 99 languages |
| Coqui XTTSv2 | TTS | GPU T4 | 8000 | Docker build — first run downloads model (~2 GB) |
| Chatterbox | TTS | GPU T4 | 8009 | Docker build — voice cloning via WAV |
| Zonos | TTS | GPU T4 | 8010 | Docker build — voice cloning via WAV |
| F5-TTS | TTS | GPU T4 | 8012 | Docker build — French fine-tune |
| VibeVoice | TTS | GPU T4 | 8020 | Docker build — experimental |

GPU engines: first startup downloads HuggingFace models and may take several minutes. Subsequent runs are much faster.

---

## Troubleshooting

**Docker permission denied**
```bash
newgrp docker   # or log out and back in
```

**nvidia-smi not found after instance start**
The NVIDIA driver comes pre-installed on Deep Learning AMIs. If using a standard Ubuntu AMI, install it:
```bash
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot
```

**Container health timeout**
GPU model containers can take 5–10 minutes on first pull. The script waits up to 600 seconds. If it still times out, check logs:
```bash
docker logs <container_name>
```

**Port already in use**
If a previous benchmark run left a container up:
```bash
docker ps
docker compose -f ~/benchmark-tts/tooling/<engine-env>/docker-compose.gpu.yml down
```

**Free disk space** (models can be large)
```bash
df -h
docker system prune -f   # removes stopped containers and dangling images
```
