#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orion Benchmark AWS — TTS / STT
=================================
Version pour instance AWS (GPU g4dn.xlarge / T4).
- Appelle les conteneurs Docker DIRECTEMENT (pas de Spring Boot)
- Lance / arrête les conteneurs un par un
- Sauvegarde le JSON dans ./results/ au format compatible orion-benchmark web
- Sauvegarde les WAVs dans ./wavs/<engine>/

Usage :
    source ~/benchmark-env/bin/activate
    python3 benchmark-aws.py
"""

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False

# ─── Chemins ──────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.resolve()
REPO_ROOT   = SCRIPT_DIR.parent.parent.resolve()
TOOLING_DIR = REPO_ROOT / "tooling"
RESULTS_DIR = SCRIPT_DIR / "results"
WAVS_DIR    = SCRIPT_DIR / "wavs"

# ─── Phrases de test ──────────────────────────────────────────────────────────
TEST_SENTENCES: list[str] = [
    #"Bonjour, je suis Orion, comment puis-je vous aider ?",
    # "Votre rendez-vous est confirmé pour demain à quatorze heures.",
    # "Vous avez deux interventions prévues cette semaine, le cinq et le sept juin.",
    # "Je n'ai pas trouvé d'informations correspondant à votre demande.",
    # "La mise à jour a été effectuée avec succès, vos données sont synchronisées.",
    # "Bienvenue chez Orion, votre assistant vocal intelligent.",
    # "Je suis désolé, je ne comprends pas votre question. Pouvez-vous reformuler ?",
    "Je n'ai pas trouvé de documentation détaillée sur le planning dans la base de connaissances disponible. Pourriez-vous me préciser ce que vous souhaitez faire ? Par exemple : Consulter le planning d'un intervenant ou d'une période ? Créer des interventions ou une série d'interventions ? Modifier ou annuler des interventions existantes ? Rechercher des intervenants disponibles sur une période ? Avec plus de détails sur votre besoin, je pourrai vous guider au mieux !"
]

# ─── Définition des moteurs ───────────────────────────────────────────────────
# Appels directs aux conteneurs Docker — pas de Spring Boot.
# health_url : GET → doit répondre HTTP < 500 quand prêt
ENGINES: dict[str, dict] = {
    "piper": {
        "type": "TTS",
        "compose_dir": "piper-env",
        "compose_file": "docker-compose.yml",   # CPU only — T4 not needed
        "port": 5000,
        "health_url": "http://localhost:5000/?text=test",
        "voices": ["default"],
        "extra": {},
        "synthesize": "piper",
        "card": {
            "id": "piper", "name": "Piper TTS", "type": "TTS",
            "hardware": "CPU", "voiceQuality": "C",
            "frenchVoices": ["fr_FR-siwis-medium"],
            "languages": ["fr", "en", "de", "es", "zh"], "languagesTotal": 35,
            "languagesNote": "Modèle .onnx séparé par langue (rebuild image)",
            "tweaks": ["Vitesse 0.5-2.0x"],
            "deploy": {"type": "docker-image", "image": "artibex/piper-http:latest", "pythonCode": False},
        },
    },
    "kokoro": {
        "type": "TTS",
        "compose_dir": "kokoro-env",
        "compose_file": "docker-compose.gpu.yml",  # GPU on AWS
        "port": 8880,
        "health_url": "http://localhost:8880/v1/audio/voices",
        "voices": ["ff_siwis"],
        "extra": {"speed": 1.0},
        "synthesize": "kokoro",
        "card": {
            "id": "kokoro", "name": "Kokoro", "type": "TTS",
            "hardware": "GPU (T4)", "voiceQuality": "A",
            "frenchVoices": ["ff_siwis"],
            "languages": ["en", "fr", "es", "it", "zh"], "languagesTotal": 9,
            "languagesNote": "Param lang_code dans la requ\u00eate",
            "tweaks": ["Vitesse 0.5-2.0x"],
            "deploy": {"type": "docker-image", "image": "ghcr.io/remsky/kokoro-fastapi-gpu:latest", "pythonCode": False},
        },
    },
    "coqui": {
        "type": "TTS",
        "compose_dir": "coqui-env",
        "compose_file": "docker-compose.gpu.yml",  # GPU on AWS
        "port": 8000,
        "health_url": "http://localhost:8000/studio_speakers",
        "voices": ["Ana Florence", "Claribel Dervla", "Abrahan Mack"],
        "extra": {"speed": 1.0},
        "synthesize": "coqui",
        "card": {
            "id": "coqui", "name": "Coqui XTTSv2", "type": "TTS",
            "hardware": "GPU (T4)", "voiceQuality": "B",
            "frenchVoices": ["Ana Florence", "Claribel Dervla", "Abrahan Mack"],
            "languages": ["fr", "en", "es", "it", "zh"], "languagesTotal": 17,
            "languagesNote": "Param language dans la requête",
            "tweaks": ["Vitesse 0.5-2.0x"],
            "deploy": {"type": "docker-build", "image": None, "buildDir": "tooling/coqui-env", "pythonCode": True},
        },
    },
    "chatterbox": {
        "type": "TTS",
        "compose_dir": "chatterbox-env",
        "compose_file": "docker-compose.gpu.yml",  # GPU on AWS
        "port": 8009,
        "health_url": "http://localhost:8009/v1/audio/voices",
        "voices": ["guillaume.wav", "orion.wav", "victoria.wav"],
        "extra": {"exaggeration": 0.5, "temperature": 0.8, "cfg_weight": 0.3},
        "synthesize": "chatterbox",
        "card": {
            "id": "chatterbox", "name": "Chatterbox", "type": "TTS",
            "hardware": "GPU (T4)", "voiceQuality": "B",
            "frenchVoices": ["guillaume", "orion", "victoria"],
            "languages": ["en", "fr", "de", "es", "zh"], "languagesTotal": 23,
            "languagesNote": "Clonage via WAV de référence",
            "tweaks": ["Exaggeration 0-2", "Temperature 0-1", "CFG Weight 0-1"],
            "deploy": {"type": "docker-build", "image": None, "buildDir": "tooling/chatterbox-env", "pythonCode": True},
        },
    },
    "f5tts": {
        "type": "TTS",
        "compose_dir": "f5tts-env",
        "compose_file": "docker-compose.gpu.yml",  # GPU on AWS
        "port": 8012,
        "health_url": "http://localhost:8012/voices",
        "voices": ["orion", "victoria"],
        "extra": {"speed": 1.0, "nfe_step": 32},
        "synthesize": "f5tts",
        "card": {
            "id": "f5tts", "name": "F5-TTS", "type": "TTS",
            "hardware": "GPU (T4)", "voiceQuality": "A",
            "frenchVoices": ["orion", "victoria"],
            "languages": ["fr"], "languagesTotal": 1,
            "languagesNote": "French fine-tune RASPIAUDIO",
            "tweaks": ["Vitesse 0.5-2.0x", "NFE steps 16-32"],
            "deploy": {"type": "docker-build", "image": None, "buildDir": "tooling/f5tts-env", "pythonCode": True},
        },
    },
    "melotts": {
        "type": "TTS",
        "compose_dir": "melotts-env",
        "compose_file": "docker-compose.gpu.yml",  # GPU on AWS
        "port": 8030,
        "health_url": "http://localhost:8030/voices",
        "voices": ["FR"],
        "extra": {"speed": 1.0},
        "synthesize": "melotts",
        "card": {
            "id": "melotts", "name": "MeloTTS", "type": "TTS",
            "hardware": "GPU (T4)", "voiceQuality": "B",
            "frenchVoices": ["FR"],
            "languages": ["fr", "en", "es", "zh", "jp", "kr"], "languagesTotal": 6,
            "languagesNote": "Mod\u00e8le l\u00e9ger, synth\u00e8se parall\u00e8le ultra-rapide",
            "tweaks": ["Vitesse 0.5-2.0x"],
            "deploy": {"type": "docker-build", "image": None, "buildDir": "tooling/melotts-env", "pythonCode": True},
        },
    },
    "parlertts": {
        "type": "TTS",
        "compose_dir": "parlertts-env",
        "compose_file": "docker-compose.gpu.yml",  # GPU on AWS
        "port": 8031,
        "health_url": "http://localhost:8031/health",
        "voices": ["female", "male"],
        "extra": {},
        "synthesize": "parlertts",
        "card": {
            "id": "parlertts", "name": "Parler-TTS", "type": "TTS",
            "hardware": "GPU (T4)", "voiceQuality": "A",
            "frenchVoices": ["female", "male"],
            "languages": ["fr", "en"], "languagesTotal": 2,
            "languagesNote": "Contr\u00f4le vocal par description texte",
            "tweaks": ["Description voix libre"],
            "deploy": {"type": "docker-build", "image": None, "buildDir": "tooling/parlertts-env", "pythonCode": True},
        },
    },
    "fishspeech": {
        "type": "TTS",
        "compose_dir": "fishspeech-env",
        "compose_file": "docker-compose.gpu.yml",
        "port": 8033,
        "health_url": "http://localhost:8033/v1/health",
        "startup_timeout": 240,
        "voices": ["default"],
        "extra": {},
        "synthesize": "fishspeech",
        "card": {
            "id": "fishspeech", "name": "Fish Speech 1.5", "type": "TTS",
            "hardware": "GPU (T4)", "voiceQuality": "A",
            "frenchVoices": ["default (al\u00e9atoire)"],
            "languages": ["fr", "en", "zh", "ja", "de", "es", "ko", "ar", "ru"], "languagesTotal": 13,
            "languagesNote": "~20k h fr, synth\u00e8se sans r\u00e9f\u00e9rence audio",
            "tweaks": ["Seed", "Temp\u00e9rature", "Clonage vocal 0-shot"],
            "deploy": {"type": "docker-build", "image": None, "buildDir": "tooling/fishspeech-env", "pythonCode": True},
        },
    },
    "magpietts": {
        "type": "TTS",
        "compose_dir": "magpietts-env",
        "compose_file": "docker-compose.gpu.yml",
        "port": 8034,
        "health_url": "http://localhost:8034/health",
        "startup_timeout": 300,
        "voices": ["Sofia", "Aria"],
        "extra": {"language": "fr"},
        "synthesize": "magpietts",
        "card": {
            "id": "magpietts", "name": "NVIDIA MagpieTTS", "type": "TTS",
            "hardware": "GPU (T4)", "voiceQuality": "A",
            "frenchVoices": ["Sofia", "Aria"],
            "languages": ["fr", "en", "es", "de", "it", "zh", "hi", "ja", "vi"], "languagesTotal": 9,
            "languagesNote": "NVIDIA NeMo 357M, test\u00e9 officiellement sur T4",
            "tweaks": ["Locuteur", "Langue", "Text normalization"],
            "deploy": {"type": "docker-build", "image": None, "buildDir": "tooling/magpietts-env", "pythonCode": True},
        },
    },    "whisper": {
        "type": "STT",
        "compose_dir": "whisper-env",
        "compose_file": "docker-compose.gpu.yml",  # GPU on AWS
        "port": 8001,
        "health_url": "http://localhost:8001/v1/models",
        "voices": ["default"],
        "extra": {},
        "synthesize": "whisper",
        "card": {
            "id": "whisper", "name": "Whisper", "type": "STT",
            "hardware": "GPU (T4)", "voiceQuality": "B",
            "frenchVoices": ["fr (langue forcée)"],
            "languages": ["fr", "en", "es", "it", "zh"], "languagesTotal": 99,
            "languagesNote": "Auto-détection ou param language",
            "tweaks": ["Modèle: tiny/base/small/medium/large-v3"],
            "deploy": {"type": "docker-image", "image": "fedirz/faster-whisper-server:latest-cuda", "pythonCode": False},
        },
    },
}

# ─── Couleurs ─────────────────────────────────────────────────────────────────
def c(text: str, color: str) -> str:
    if not HAS_COLOR:
        return text
    colors = {
        "blue": Fore.BLUE, "cyan": Fore.CYAN, "green": Fore.GREEN,
        "yellow": Fore.YELLOW, "red": Fore.RED, "white": Fore.WHITE,
        "bold": Style.BRIGHT, "dim": Style.DIM,
    }
    return colors.get(color, "") + text + Style.RESET_ALL

def sep(char: str = "─", n: int = 60) -> str:
    return c(char * n, "dim")

def fmt_ms(ms: float) -> str:
    if ms <= 0:     return "—"
    if ms < 1_000:  return f"{ms:.0f}ms"
    if ms < 60_000: return f"{ms / 1_000:.1f}s"
    return f"{ms / 60_000:.1f}min"

# ─── Hardware detection ───────────────────────────────────────────────────────
def detect_hardware() -> tuple[str, str, str]:
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                timeout=5, text=True
            ).strip().splitlines()
            if out:
                model = out[0].strip()
                return "GPU", model, f"NVIDIA {model}"
        except Exception:
            pass
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    cpu = line.split(":")[1].strip()
                    break
    except Exception:
        pass
    return "CPU", cpu, cpu

# ─── Docker helpers ───────────────────────────────────────────────────────────
def stop_all_containers() -> None:
    """Kill every running container before starting a benchmark engine."""
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.ID}}\t{{.Names}}"],
        capture_output=True, text=True
    )
    lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    if not lines:
        print(f"  {c('✓ Aucun conteneur actif', 'dim')}")
        return
    print(f"  {c('⚠', 'yellow')} {len(lines)} conteneur(s) actif(s) détecté(s) — arrêt en cours…")
    for line in lines:
        cid, *name_parts = line.split("\t")
        name = name_parts[0] if name_parts else cid
        subprocess.run(["docker", "stop", cid], capture_output=True)
        print(f"  {c('■', 'dim')} Arrêté : {name}")

def compose_up(engine_id: str) -> bool:
    cfg = ENGINES[engine_id]
    compose_dir = TOOLING_DIR / cfg["compose_dir"]
    compose_file = cfg["compose_file"]
    print(f"  {c('▶', 'cyan')} Démarrage {c(engine_id, 'bold')} ({compose_file}) …", flush=True)
    cmd = ["docker", "compose", "-f", str(compose_dir / compose_file), "up", "-d"]
    # Stream output line by line so the user sees build progress
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if line:
            print(f"  {c('│', 'dim')} {line}")
    process.wait()
    if process.returncode != 0:
        print(c(f"  ✗ docker compose up a échoué (code {process.returncode})", "red"))
        return False
    return True

def compose_down(engine_id: str) -> None:
    cfg = ENGINES[engine_id]
    compose_dir = TOOLING_DIR / cfg["compose_dir"]
    compose_file = cfg["compose_file"]
    print(f"  {c('■', 'dim')} Arrêt {engine_id} …", flush=True)
    cmd = ["docker", "compose", "-f", str(compose_dir / compose_file), "down"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout.strip():
        for line in result.stdout.strip().splitlines():
            print(f"  {c('│', 'dim')} {line}")
    print(f"  {c('■', 'dim')} {engine_id} arrêté")

def wait_ready(engine_id: str, timeout: int = 600) -> bool:
    url = ENGINES[engine_id]["health_url"]
    t_start = time.time()
    print(f"  {c('⏳', 'yellow')} En attente de {c(engine_id, 'bold')} ({url}) …", flush=True)
    while True:
        elapsed = int(time.time() - t_start)
        if elapsed >= timeout:
            print(c(f"  ✗ Timeout ({timeout}s)", "red"))
            return False
        try:
            r = requests.get(url, timeout=(5, 90))
            if r.status_code < 500:
                print(f"  {c('✓ Prêt', 'green')} ({elapsed}s, HTTP {r.status_code})")
                return True
            print(f"  ... {elapsed}s — HTTP {r.status_code}", flush=True)
        except requests.exceptions.ConnectionError:
            print(f"  ... {elapsed}s — connexion refusée (démarrage en cours) …", flush=True)
        except requests.exceptions.Timeout:
            print(f"  ... {elapsed}s — pas de réponse (modèle en cours de chargement) …", flush=True)
        except Exception as e:
            print(f"  ... {elapsed}s — {str(e)[:80]}", flush=True)
        time.sleep(8)

# ─── Synthesis — appel direct aux conteneurs Docker ──────────────────────────
_COQUI_SPEAKERS: dict = {}

def _get_coqui_speaker(voice: str) -> tuple[list, list] | None:
    global _COQUI_SPEAKERS
    if not _COQUI_SPEAKERS:
        try:
            r = requests.get("http://localhost:8000/studio_speakers", timeout=15)
            if r.status_code == 200:
                _COQUI_SPEAKERS = r.json()
        except Exception:
            return None
    spk = _COQUI_SPEAKERS.get(voice)
    if spk:
        return spk["speaker_embedding"], spk["gpt_cond_latent"]
    return None

def synthesize(engine_id: str, text: str, voice: str, timeout: float) -> tuple[float, bytes | None, str | None]:
    cfg = ENGINES[engine_id]
    extra = cfg["extra"]
    mode = cfg["synthesize"]
    t0 = time.perf_counter()

    try:
        if mode == "piper":
            r = requests.get("http://localhost:5000/", params={"text": text}, timeout=(5, timeout))

        elif mode == "kokoro":
            r = requests.post("http://localhost:8880/v1/audio/speech", json={
                "model": "kokoro", "input": text, "voice": voice,
                "response_format": "wav", "speed": extra.get("speed", 1.0),
            }, timeout=(5, timeout))

        elif mode == "coqui":
            spk = _get_coqui_speaker(voice)
            if spk is None:
                return 0.0, None, f"Speaker '{voice}' introuvable"
            emb, latent = spk
            r = requests.post("http://localhost:8000/tts_stream", json={
                "text": text, "language": "fr",
                "speaker_embedding": emb, "gpt_cond_latent": latent,
                "add_wav_header": True, "stream_chunk_size": "20",
                "speed": extra.get("speed", 1.0),
            }, timeout=(5, timeout))

        elif mode == "chatterbox":
            r = requests.post("http://localhost:8009/v1/audio/stream", json={
                "input": text, "language": "fr", "voice_ref": voice,
                "exaggeration": extra.get("exaggeration", 0.5),
                "temperature": extra.get("temperature", 0.8),
                "cfg_weight": extra.get("cfg_weight", 0.3),
            }, timeout=(5, timeout))

        elif mode == "f5tts":
            r = requests.post("http://localhost:8012/tts", json={
                "text": text, "voice": voice,
                "speed": extra.get("speed", 1.0),
                "nfe_step": extra.get("nfe_step", 32),
            }, timeout=(5, timeout))

        elif mode == "melotts":
            r = requests.post("http://localhost:8030/tts", json={
                "text": text, "voice": voice, "speed": extra.get("speed", 1.0),
            }, timeout=(5, timeout))


        elif mode == "fishspeech":
            # Fish Speech v1.5 API: POST /v1/tts with JSON body
            # voice parameter is not used (random voice synthesis)
            r = requests.post("http://localhost:8033/v1/tts",
                json={"text": text, "format": "wav", "streaming": False},
                headers={"Content-Type": "application/json"},
                timeout=(5, timeout))

        elif mode == "magpietts":
            r = requests.post("http://localhost:8034/tts", json={
                "text": text, "voice": voice, "language": extra.get("language", "fr"),
            }, timeout=(5, timeout))

        elif mode == "whisper":
            return 0.0, None, "STT engine — pas de synthèse"

        else:
            return 0.0, None, f"Engine '{engine_id}' non géré"

        lat = (time.perf_counter() - t0) * 1_000
        if r.status_code == 200:
            return lat, r.content, None
        return lat, None, f"HTTP {r.status_code}: {r.text[:200]}"

    except requests.exceptions.Timeout:
        return timeout * 1_000, None, f"Timeout après {timeout:.0f}s"
    except Exception as e:
        return 0.0, None, str(e)

# ─── Benchmark d'un moteur ────────────────────────────────────────────────────
def benchmark_engine(
    engine_id: str, runs_per_sentence: int, timeout: float, wav_dir: Path, run_ts: str,
    voices_override: list[str] | None = None,
) -> tuple[list[dict], list[float]]:
    cfg = ENGINES[engine_id]
    voices = voices_override if voices_override is not None else cfg["voices"]
    runs: list[dict] = []
    latencies: list[float] = []

    total = len(voices) * len(TEST_SENTENCES) * runs_per_sentence
    done = 0

    for voice in voices:
        for si, sentence in enumerate(TEST_SENTENCES):
            for ri in range(runs_per_sentence):
                done += 1
                short = sentence[:45] + "…" if len(sentence) > 45 else sentence
                voice_label = voice.replace(".wav", "")
                print(f"  [{done:>3}/{total}] {c(voice_label, 'cyan')} · phrase {si+1} · run {ri+1}  ", end="", flush=True)

                lat_ms, audio, err = synthesize(engine_id, sentence, voice, timeout)
                wav_name = f"{run_ts}_{engine_id}_{voice_label}_s{si:02d}_r{ri}.wav"

                record: dict = {
                    "engine": engine_id,
                    "voice": voice_label,
                    "sentenceIdx": si,
                    "sentence": sentence,
                    "runIdx": ri,
                    "latencyMs": round(lat_ms, 1),
                    "success": audio is not None,
                    "errorMsg": err,
                    "wavFile": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                if audio:
                    wav_path = wav_dir / wav_name
                    wav_path.write_bytes(audio)
                    record["wavFile"] = f"wavs/{engine_id}/{wav_name}"
                    latencies.append(lat_ms)
                    print(c(f"→ {fmt_ms(lat_ms)} ✓", "green"))
                else:
                    print(c(f"→ ERREUR: {err}", "red"))

                runs.append(record)

    return runs, latencies

# ─── JSON session ─────────────────────────────────────────────────────────────
def load_or_create_session(json_path: Path, hw_type: str, hw_model: str, hardware: str) -> dict:
    if json_path.exists():
        try:
            return json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "label": f"{hw_type} — {hw_model}",
        "meta": {
            "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hardware": hardware,
            "hwType": hw_type,
            "hwModel": hw_model,
            "runsPerSentence": 0,
            "sentences": TEST_SENTENCES,
            "note": "Généré par orion-benchmark/benchmark-aws.py (AWS GPU)",
        },
        "ttsEngines": [],
        "sttEngines": [],
        "runs": [],
    }

def upsert_engine_card(session: dict, engine_id: str, latencies: list[float]) -> None:
    cfg = ENGINES[engine_id]
    card = dict(cfg["card"])
    if latencies:
        avg = round(sum(latencies) / len(latencies))
        card["latencyMs"] = avg
        card["latencyLabel"] = fmt_ms(avg)
        card["benchmarkRuns"] = len(latencies)
    else:
        card["latencyMs"] = 0
        card["latencyLabel"] = "—"
        card["benchmarkRuns"] = 0
    key = "ttsEngines" if cfg["type"] == "TTS" else "sttEngines"
    lst: list = session[key]
    idx = next((i for i, e in enumerate(lst) if e.get("id") == engine_id), None)
    if idx is not None:
        lst[idx] = card
    else:
        lst.append(card)

def save_session(session: dict, json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")

# ─── CLI ──────────────────────────────────────────────────────────────────────
def print_header(hw_type: str, hw_model: str) -> None:
    print()
    print(c("  ╔══════════════════════════════════════════╗", "blue"))
    print(c("  ║   ", "blue") + c("ORION BENCHMARK AWS — TTS / STT", "bold") + c("   ║", "blue"))
    print(c("  ╚══════════════════════════════════════════╝", "blue"))
    print(f"  {c('Matériel :', 'dim')} {c(hw_type, 'cyan')} — {hw_model}")
    print()

def choose_engines() -> list[str]:
    tts_ids = [k for k, v in ENGINES.items() if v["type"] == "TTS"]
    stt_ids = [k for k, v in ENGINES.items() if v["type"] == "STT"]
    all_ids = tts_ids + stt_ids

    print(sep())
    print(c("  Moteurs disponibles :", "bold"))
    print()
    # header
    print(f"  {'':3s}  {'Nom':<18s}  {'Qualité':<12s}  {'HW':<10s}  {'Voix (total)'}")
    print(f"  {sep('-', 70)}")
    print(c("  TTS", "cyan"))
    for i, eid in enumerate(tts_ids, 1):
        card = ENGINES[eid]["card"]
        compose = ENGINES[eid]["compose_file"]
        gpu_tag = c("GPU", "yellow") if "gpu" in compose else c("CPU", "dim")
        voices = ENGINES[eid]["voices"]
        voice_labels = ", ".join(v.replace(".wav", "") for v in voices)
        voice_count = c(f"({len(voices)})", "dim")
        print(f"  {c(str(i), 'yellow'):>3s}  {c(card['name'], 'bold'):<28s}  {card['voiceQuality']+'-quality':<12s}  {gpu_tag:<18s}  {voice_labels} {voice_count}")
    print()
    print(c("  STT", "cyan"))
    for i, eid in enumerate(stt_ids, len(tts_ids) + 1):
        card = ENGINES[eid]["card"]
        voices = ENGINES[eid]["voices"]
        voice_labels = ", ".join(v.replace(".wav", "") for v in voices)
        voice_count = c(f"({len(voices)})", "dim")
        print(f"  {c(str(i), 'yellow'):>3s}  {c(card['name'], 'bold'):<28s}  {card['voiceQuality']+'-quality':<12s}  {c('GPU', 'yellow'):<18s}  {voice_labels} {voice_count}")
    print()
    print(f"  {c('*', 'yellow')}  {c('Tous les moteurs', 'bold')}")
    print()
    print(sep())

    while True:
        raw = input(c("  Choix (numéro, virgules, ou *) : ", "bold")).strip()
        if raw == "*":
            return all_ids
        selected: list[str] = []
        valid = True
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit():
                idx = int(part) - 1
                if 0 <= idx < len(all_ids):
                    selected.append(all_ids[idx])
                else:
                    print(c(f"  Numéro invalide : {part}", "red")); valid = False; break
            elif part in all_ids:
                selected.append(part)
            else:
                print(c(f"  Moteur inconnu : {part}", "red")); valid = False; break
        if valid and selected:
            return selected
        elif valid:
            print(c("  Sélection vide, réessayez.", "red"))

def choose_voices(selected_engines: list[str]) -> dict[str, list[str]]:
    """For each engine that has multiple voices, ask the user which to use."""
    voice_selection: dict[str, list[str]] = {}
    for eid in selected_engines:
        all_voices = ENGINES[eid]["voices"]
        if len(all_voices) <= 1:
            voice_selection[eid] = all_voices
            continue
        print()
        print(sep())
        print(c(f"  Voix disponibles pour {c(ENGINES[eid]['card']['name'], 'bold')} :", "bold"))
        print()
        for i, v in enumerate(all_voices, 1):
            label = v.replace(".wav", "")
            print(f"  {c(str(i), 'yellow')}  {label}")
        print()
        print(f"  {c('*', 'yellow')}  Toutes les voix")
        print()
        print(sep())
        while True:
            raw = input(c(f"  Voix pour {eid} (numéro, virgules, ou *) : ", "bold")).strip()
            if raw == "*":
                voice_selection[eid] = all_voices
                break
            selected: list[str] = []
            valid = True
            for part in raw.split(","):
                part = part.strip()
                if part.isdigit():
                    idx = int(part) - 1
                    if 0 <= idx < len(all_voices):
                        selected.append(all_voices[idx])
                    else:
                        print(c(f"  Numéro invalide : {part}", "red")); valid = False; break
                else:
                    print(c(f"  Entrée invalide : {part}", "red")); valid = False; break
            if valid and selected:
                voice_selection[eid] = selected
                break
            elif valid:
                print(c("  Sélection vide, réessayez.", "red"))
    return voice_selection

def choose_runs() -> int:
    raw = input(c("  Runs par phrase (défaut 3) : ", "bold")).strip()
    if not raw:
        return 3
    try:
        return max(1, int(raw))
    except ValueError:
        return 3

def choose_output(hw_type: str, hw_model: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = [p for p in sorted(RESULTS_DIR.glob("*.json")) if p.name != "index.json"]

    safe_model = hw_model.replace(" ", "-").replace("/", "-")[:30]
    default_name = f"benchmark-aws-{hw_type.lower()}-{safe_model}.json"

    print()
    print(sep())
    print(c("  Fichier de résultats :", "bold"))
    print()

    if existing:
        print(c("  Fichiers existants :", "dim"))
        for i, p in enumerate(existing, 1):
            size = p.stat().st_size // 1024
            print(f"  {c(str(i), 'yellow')}  {p.name}  {c(f'({size} KB)', 'dim')}")
        print()

    print(f"  {c('N', 'yellow')}  Nouveau fichier")
    print()

    while True:
        raw = input(c("  Choix (numéro ou N) : ", "bold")).strip()
        if raw.isdigit():
            idx = int(raw) - 1
            if existing and 0 <= idx < len(existing):
                chosen = existing[idx]
                print(f"  {c('→ Append vers', 'dim')} {c(chosen.name, 'cyan')}")
                return chosen
            print(c("  Numéro invalide.", "red"))
            continue
        if raw.upper() == "N":
            name_raw = input(c(f"  Nom du fichier [{default_name}] : ", "bold")).strip()
            if not name_raw:
                name_raw = default_name
            if not name_raw.endswith(".json"):
                name_raw += ".json"
            new_path = RESULTS_DIR / name_raw
            if new_path.exists():
                ow = input(c(f"  ⚠ {name_raw} existe. Écraser ? [o/N] : ", "yellow")).strip().lower()
                if ow not in ("o", "y", "oui", "yes"):
                    continue
            print(f"  {c('→ Nouveau :', 'dim')} {c(name_raw, 'cyan')}")
            return new_path
        print(c("  Entrez un numéro ou N.", "red"))

# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    hw_type, hw_model, hardware_full = detect_hardware()
    print_header(hw_type, hw_model)

    selected_engines = choose_engines()
    voice_selection   = choose_voices(selected_engines)
    runs_per_sentence = choose_runs()
    json_path = choose_output(hw_type, hw_model)

    # WAVs per engine subfolder
    timeout = 300.0

    print()
    print(sep())
    print(c("  Récapitulatif :", "bold"))
    print(f"  Moteurs  : {c(', '.join(selected_engines), 'cyan')}")
    voices_summary = " | ".join(
        f"{eid}: {', '.join(v.replace('.wav','') for v in voice_selection.get(eid, ENGINES[eid]['voices']))}"
        for eid in selected_engines
    )
    print(f"  Voix     : {c(voices_summary, 'cyan')}")
    print(f"  Runs     : {c(str(runs_per_sentence), 'yellow')} × {len(TEST_SENTENCES)} phrases")
    total_req = sum(
        len(voice_selection.get(e, ENGINES[e]["voices"])) * len(TEST_SENTENCES) * runs_per_sentence
        for e in selected_engines
    )
    print(f"  Requêtes : {c(str(total_req), 'yellow')} au total")
    print(f"  Sortie   : {c(str(json_path), 'dim')}")
    print(sep())
    print()

    session = load_or_create_session(json_path, hw_type, hw_model, hardware_full)
    session["meta"]["runsPerSentence"] = runs_per_sentence
    session["meta"]["sentences"] = TEST_SENTENCES

    all_results: dict[str, tuple[list[dict], list[float]]] = {}

    # ── Pre-flight: stop any running containers ────────────────────────────
    print()
    print(sep())
    print(c("  Vérification des conteneurs actifs…", "bold"))
    stop_all_containers()
    print(sep())

    for engine_id in selected_engines:
        print()
        print(sep("═"))
        print(c(f"  ▶ {ENGINES[engine_id]['card']['name']} ({engine_id})", "bold"))
        print(sep("═"))

        # Start container
        ok = compose_up(engine_id)
        if not ok:
            choice = input(c("  [I]gnorer  [R]éessayer : ", "yellow")).strip().lower()
            if choice == "r":
                ok = compose_up(engine_id)
            if not ok:
                print(c(f"  ✗ {engine_id} ignoré.", "dim"))
                continue

        # Wait until ready
        ready = wait_ready(engine_id, timeout=ENGINES[engine_id].get("startup_timeout", 600))
        if not ready:
            choice = input(c("  [I]gnorer  [R]éessayer l'attente : ", "yellow")).strip().lower()
            if choice == "r":
                ready = wait_ready(engine_id, timeout=ENGINES[engine_id].get("startup_timeout", 600))
            if not ready:
                compose_down(engine_id)
                print(c(f"  ✗ {engine_id} ignoré.", "dim"))
                continue

        # Run benchmark
        wav_dir = WAVS_DIR / engine_id
        wav_dir.mkdir(parents=True, exist_ok=True)
        run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        print(f"\n  {c('Mesures en cours…', 'bold')}")
        runs, latencies = benchmark_engine(engine_id, runs_per_sentence, timeout, wav_dir, run_ts,
                                            voices_override=voice_selection.get(engine_id))
        all_results[engine_id] = (runs, latencies)

        if latencies:
            avg = sum(latencies) / len(latencies)
            print(f"\n  {c('Résultat :', 'bold')} moy. {c(fmt_ms(avg), 'green')}  "
                  f"min {c(fmt_ms(min(latencies)), 'cyan')}  "
                  f"max {c(fmt_ms(max(latencies)), 'yellow')}")

        # Stop container
        compose_down(engine_id)

        # Save incrementally (Ctrl+C safe)
        new_engine_ids = set(all_results.keys())
        kept_runs = [r for r in session.get("runs", []) if r.get("engine") not in new_engine_ids]
        session["runs"] = kept_runs + [r for e_runs, _ in all_results.values() for r in e_runs]
        for eid, (_, lats) in all_results.items():
            upsert_engine_card(session, eid, lats)
        save_session(session, json_path)
        print(c(f"  ✓ Sauvegardé → {json_path.name}", "green"))

    # Final summary
    print()
    print(sep("═"))
    print(c("  RÉSULTATS FINAUX", "bold"))
    print(sep("═"))
    print(f"  {'Moteur':16s}  {'Moy.':10s}  {'Min':10s}  {'Max':10s}  Runs")
    print(sep())
    for eid, (_, lats) in all_results.items():
        name = ENGINES[eid]["card"]["name"]
        if lats:
            print(f"  {name:16s}  "
                  f"{c(fmt_ms(sum(lats)/len(lats)), 'green'):20s}  "
                  f"{fmt_ms(min(lats)):10s}  "
                  f"{fmt_ms(max(lats)):10s}  {len(lats)}")
        else:
            print(f"  {name:16s}  {c('indisponible', 'red')}")
    print(sep())
    print()
    print(f"  {c('✓ JSON  :', 'green')} {json_path}")
    print(f"  {c('✓ WAVs  :', 'green')} {WAVS_DIR}/<engine>/")
    print()
    print(c("  Benchmark terminé !", "bold"))
    print()
    print(c("  Pour récupérer le JSON :", "dim"))
    print(f"  cat {json_path}")
    print()


if __name__ == "__main__":
    main()
