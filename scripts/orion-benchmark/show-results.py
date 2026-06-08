#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orion Benchmark — Results Viewer
==================================
Reads a benchmark JSON and prints a human-readable summary table.

Usage:
    python3 show-results.py                     # pick from list
    python3 show-results.py results/my-file.json
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False

SCRIPT_DIR  = Path(__file__).parent.resolve()
RESULTS_DIR = SCRIPT_DIR / "results"

# ─── Color helpers ────────────────────────────────────────────────────────────
def c(text: str, color: str) -> str:
    if not HAS_COLOR:
        return text
    colors = {
        "blue": Fore.BLUE, "cyan": Fore.CYAN, "green": Fore.GREEN,
        "yellow": Fore.YELLOW, "red": Fore.RED, "white": Fore.WHITE,
        "bold": Style.BRIGHT, "dim": Style.DIM, "magenta": Fore.MAGENTA,
    }
    return colors.get(color, "") + text + Style.RESET_ALL

def sep(char: str = "─", n: int = 90) -> str:
    return c(char * n, "dim")

def fmt_ms(ms: float) -> str:
    if ms <= 0:     return c("—", "dim")
    if ms < 1_000:  return f"{ms:.0f}ms"
    if ms < 60_000: return f"{ms / 1_000:.1f}s"
    return f"{ms / 60_000:.1f}min"

def fmt_ms_plain(ms: float) -> str:
    if ms <= 0:     return "—"
    if ms < 1_000:  return f"{ms:.0f}ms"
    if ms < 60_000: return f"{ms / 1_000:.1f}s"
    return f"{ms / 60_000:.1f}min"

# ─── File picker ──────────────────────────────────────────────────────────────
def pick_file() -> Path:
    files = sorted(p for p in RESULTS_DIR.glob("*.json") if p.name != "index.json")
    if not files:
        print(c(f"  Aucun fichier JSON trouvé dans {RESULTS_DIR}", "red"))
        sys.exit(1)

    print()
    print(sep())
    print(c("  Fichiers de résultats disponibles :", "bold"))
    print()
    for i, p in enumerate(files, 1):
        size = p.stat().st_size // 1024
        print(f"  {c(str(i), 'yellow')}  {p.name}  {c(f'({size} KB)', 'dim')}")
    print()
    print(sep())

    while True:
        raw = input(c("  Choix (numéro) : ", "bold")).strip()
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(files):
                return files[idx]
        print(c("  Numéro invalide.", "red"))

# ─── Main display ─────────────────────────────────────────────────────────────
def show(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))

    meta        = data.get("meta", {})
    runs_all    = data.get("runs", [])
    tts_engines = data.get("ttsEngines", [])
    stt_engines = data.get("sttEngines", [])

    # ── Header ────────────────────────────────────────────────────────────────
    print()
    print(c("  ╔══════════════════════════════════════════════════════╗", "blue"))
    print(c("  ║   ", "blue") + c("ORION BENCHMARK — RÉSULTATS", "bold") + c("                       ║", "blue"))
    print(c("  ╚══════════════════════════════════════════════════════╝", "blue"))
    print(f"  {c('Fichier  :', 'dim')} {c(path.name, 'cyan')}")
    print(f"  {c('Matériel :', 'dim')} {c(meta.get('hardware', '—'), 'cyan')}")
    print(f"  {c('Généré   :', 'dim')} {meta.get('generatedAt', '—')}")
    runs_per = meta.get('runsPerSentence', '—')
    sentences = meta.get('sentences', [])
    print(f"  {c('Runs/phrase :', 'dim')} {runs_per}   {c('Phrases testées :', 'dim')} {len(sentences)}")
    for i, s in enumerate(sentences, 1):
        short = s[:110] + "…" if len(s) > 110 else s
        print(f"  {c(str(i) + '.', 'dim')} {c(short, 'white')}")
    print()

    # ── Build per-engine stats from raw runs ───────────────────────────────────
    engine_stats: dict[str, dict] = defaultdict(lambda: {
        "latencies": [], "errors": 0, "voices": set(), "sentences": set()
    })
    for run in runs_all:
        eid  = run.get("engine", "?")
        lat  = run.get("latencyMs", 0)
        ok   = run.get("success", False)
        voice = run.get("voice", "?")
        sidx  = run.get("sentenceIdx", 0)
        if ok and lat > 0:
            engine_stats[eid]["latencies"].append(lat)
        else:
            engine_stats[eid]["errors"] += 1
        engine_stats[eid]["voices"].add(voice)
        engine_stats[eid]["sentences"].add(sidx)

    def print_engine_table(engines: list[dict], section_label: str) -> None:
        if not engines:
            return
        print(sep("═"))
        print(c(f"  {section_label}", "bold"))
        print(sep("═"))
        print()

        # Column widths
        COL = {"name": 16, "hw": 10, "voices": 30, "runs": 6, "ok": 5,
               "avg": 9, "min": 9, "max": 9, "err": 5}

        header = (
            f"  {c('Moteur', 'bold'):<{COL['name']+10}}  "
            f"{'HW':<{COL['hw']}}  "
            f"{'Voix testées':<{COL['voices']}}  "
            f"{'Runs':>{COL['runs']}}  "
            f"{'OK':>{COL['ok']}}  "
            f"{'Moy.':>{COL['avg']}}  "
            f"{'Min':>{COL['min']}}  "
            f"{'Max':>{COL['max']}}  "
            f"{'Err':>{COL['err']}}"
        )
        print(header)
        print(f"  {sep('-', 88)}")

        for card in engines:
            eid    = card.get("id", "?")
            name   = card.get("name", eid)
            hw     = card.get("hardware", "?")
            stats  = engine_stats.get(eid, {})
            lats   = stats.get("latencies", [])
            errors = stats.get("errors", 0)
            voices_tested = stats.get("voices", set())
            total_runs = len(lats) + errors

            if lats:
                avg_ms = sum(lats) / len(lats)
                min_ms = min(lats)
                max_ms = max(lats)
                avg_str = c(fmt_ms_plain(avg_ms), "green")
                min_str = c(fmt_ms_plain(min_ms), "cyan")
                max_str = c(fmt_ms_plain(max_ms), "yellow")
                ok_str  = c(str(len(lats)), "green")
            else:
                avg_str = c("—", "dim")
                min_str = c("—", "dim")
                max_str = c("—", "dim")
                ok_str  = c("0", "red")

            hw_col  = c(hw, "yellow") if "GPU" in hw else c(hw, "dim")
            err_str = c(str(errors), "red") if errors else c("0", "dim")
            voices_str = ", ".join(sorted(voices_tested)) if voices_tested else c("—", "dim")
            if len(voices_str) > COL["voices"]:
                voices_str = voices_str[:COL["voices"] - 1] + "…"

            print(
                f"  {c(name, 'bold'):<{COL['name']+10}}  "
                f"{hw_col:<{COL['hw']+10}}  "
                f"{voices_str:<{COL['voices']}}  "
                f"{str(total_runs):>{COL['runs']}}  "
                f"{ok_str:>{COL['ok']+10}}  "
                f"{avg_str:>{COL['avg']+10}}  "
                f"{min_str:>{COL['min']+10}}  "
                f"{max_str:>{COL['max']+10}}  "
                f"{err_str:>{COL['err']+10}}"
            )

            # Per-voice breakdown if more than one voice
            if len(voices_tested) > 1:
                for voice in sorted(voices_tested):
                    v_lats = [
                        r["latencyMs"] for r in runs_all
                        if r.get("engine") == eid
                        and r.get("voice") == voice
                        and r.get("success")
                        and r.get("latencyMs", 0) > 0
                    ]
                    v_err = sum(
                        1 for r in runs_all
                        if r.get("engine") == eid
                        and r.get("voice") == voice
                        and not r.get("success")
                    )
                    if v_lats:
                        v_avg = fmt_ms_plain(sum(v_lats) / len(v_lats))
                        v_min = fmt_ms_plain(min(v_lats))
                        v_max = fmt_ms_plain(max(v_lats))
                        v_ok  = str(len(v_lats))
                    else:
                        v_avg = v_min = v_max = "—"
                        v_ok  = "0"
                    v_err_s = str(v_err) if v_err else "—"
                    print(
                        f"  {c('  └ ' + voice, 'dim'):<{COL['name']+14}}  "
                        f"{'':>{COL['hw']}}  "
                        f"{'':>{COL['voices']}}  "
                        f"{str(len(v_lats) + v_err):>{COL['runs']}}  "
                        f"{c(v_ok, 'green'):>{COL['ok']+10}}  "
                        f"{c(v_avg, 'cyan'):>{COL['avg']+10}}  "
                        f"{c(v_min, 'cyan'):>{COL['min']+10}}  "
                        f"{c(v_max, 'cyan'):>{COL['max']+10}}  "
                        f"{c(v_err_s, 'red') if v_err else c('—', 'dim'):>{COL['err']+10}}"
                    )
        print()

    print_engine_table(tts_engines, "TTS — Text to Speech")
    print_engine_table(stt_engines, "STT — Speech to Text")

    # ── Overall summary ────────────────────────────────────────────────────────
    all_lats = [r["latencyMs"] for r in runs_all if r.get("success") and r.get("latencyMs", 0) > 0]
    all_errs = sum(1 for r in runs_all if not r.get("success"))
    print(sep())
    print(c("  RÉSUMÉ GLOBAL", "bold"))
    print(sep())
    print(f"  Total requêtes : {c(str(len(runs_all)), 'cyan')}   "
          f"Succès : {c(str(len(all_lats)), 'green')}   "
          f"Erreurs : {c(str(all_errs), 'red' if all_errs else 'dim')}")
    if all_lats:
        print(f"  Latence globale : moy {c(fmt_ms(sum(all_lats)/len(all_lats)), 'green')}  "
              f"min {c(fmt_ms(min(all_lats)), 'cyan')}  "
              f"max {c(fmt_ms(max(all_lats)), 'yellow')}")
    engines_done = list({r["engine"] for r in runs_all})
    print(f"  Moteurs benchmarkés : {c(str(len(engines_done)), 'cyan')} "
          f"({', '.join(engines_done)})")
    print(sep())
    print()


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if not path.is_absolute():
            path = SCRIPT_DIR / path
        if not path.exists():
            print(c(f"  Fichier introuvable : {path}", "red"))
            sys.exit(1)
    else:
        path = pick_file()

    show(path)
