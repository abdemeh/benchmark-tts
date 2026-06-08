#!/usr/bin/env bash
# =============================================================================
#  Orion Benchmark — AWS g4dn.xlarge (Ubuntu) Setup Script
# =============================================================================
#  Run once on a fresh Ubuntu instance to install all dependencies,
#  configure Docker + NVIDIA GPU support, and prepare the Python env.
#
#  Usage:
#    chmod +x setup-aws.sh
#    ./setup-aws.sh
# =============================================================================
set -euo pipefail

REPO_URL="https://github.com/abdemeh/benchmark-tts.git"
REPO_DIR="$HOME/benchmark-tts"
VENV_DIR="$HOME/benchmark-env"

GREEN="\033[0;32m"; YELLOW="\033[1;33m"; RED="\033[0;31m"; RESET="\033[0m"
info()  { echo -e "${GREEN}[INFO]${RESET}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error() { echo -e "${RED}[ERROR]${RESET} $*"; exit 1; }

# ─── 1. System packages ───────────────────────────────────────────────────────
info "Updating apt packages…"
sudo apt-get update -qq
sudo apt-get install -y \
    git curl wget ca-certificates gnupg lsb-release \
    python3 python3-pip python3-venv \
    ffmpeg \
    --no-install-recommends

# ─── 2. Docker ────────────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    info "Installing Docker…"
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] \
        https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
        | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    sudo systemctl enable --now docker
    sudo usermod -aG docker "$USER"
    info "Docker installed. NOTE: you may need to log out/in for group changes to take effect."
else
    info "Docker already installed: $(docker --version)"
fi

# ─── 3. NVIDIA Container Toolkit (GPU passthrough) ────────────────────────────
if ! dpkg -l | grep -q nvidia-container-toolkit 2>/dev/null; then
    info "Installing NVIDIA Container Toolkit…"
    distribution=$(. /etc/os-release; echo "$ID$VERSION_ID")
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -sL "https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list" \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update -qq
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    info "NVIDIA Container Toolkit installed."
else
    info "NVIDIA Container Toolkit already installed."
fi

# ─── 4. Verify GPU ────────────────────────────────────────────────────────────
info "GPU check:"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    warn "nvidia-smi not found — make sure the NVIDIA driver is installed on this instance."
fi

# ─── 5. Clone / update the repo ───────────────────────────────────────────────
if [ ! -d "$REPO_DIR" ]; then
    info "Cloning $REPO_URL → $REPO_DIR"
    git clone "$REPO_URL" "$REPO_DIR"
else
    info "Repo already exists at $REPO_DIR — pulling latest…"
    git -C "$REPO_DIR" pull
fi

# ─── 6. Python virtual environment ────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    info "Creating Python venv at $VENV_DIR…"
    python3 -m venv "$VENV_DIR"
fi
info "Installing Python dependencies…"
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -r "$REPO_DIR/scripts/orion-benchmark/requirements.txt" -q

# ─── 7. Pre-pull public Docker images (saves time later) ──────────────────────
info "Pre-pulling public Docker images (piper, kokoro, whisper)…"
# Run as current user — if not yet in docker group use sudo
DOCKER="docker"
if ! docker ps &>/dev/null; then
    DOCKER="sudo docker"
    warn "Using sudo for docker commands (not yet in docker group)."
fi

$DOCKER pull artibex/piper-http:latest || warn "piper pull failed — will build on first run"
$DOCKER pull ghcr.io/remsky/kokoro-fastapi-cpu:latest || warn "kokoro pull failed"
$DOCKER pull fedirz/faster-whisper-server:latest-cpu || warn "whisper pull failed"

# ─── 8. Done ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}============================================================${RESET}"
echo -e "${GREEN}  Setup complete!${RESET}"
echo -e "${GREEN}============================================================${RESET}"
echo ""
echo -e "  Activate the venv and run the benchmark:"
echo -e "  ${YELLOW}source $VENV_DIR/bin/activate${RESET}"
echo -e "  ${YELLOW}cd $REPO_DIR/scripts/orion-benchmark${RESET}"
echo -e "  ${YELLOW}python3 benchmark-aws.py${RESET}"
echo ""
echo -e "  To retrieve results afterwards:"
echo -e "  ${YELLOW}cat $REPO_DIR/scripts/orion-benchmark/results/<your-file>.json${RESET}"
echo ""
if groups | grep -qv docker; then
    warn "Remember to run 'newgrp docker' or re-login so your user can run docker without sudo."
fi
