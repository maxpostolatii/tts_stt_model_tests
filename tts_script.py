#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import subprocess
from pathlib import Path

# -------- Backends --------
def tts_en_coqui(text: str, out_path: Path, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"):
    from TTS.api import TTS
    out_path.parent.mkdir(parents=True, exist_ok=True)
    TTS(model_name=model_name).tts_to_file(text=text, file_path=str(out_path))

def tts_uk_espeak(text: str, out_path: Path, voice: str = "uk", rate: int = 170, pitch: int = 50, volume: int = 200):

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "espeak-ng",
        "-v", voice,
        "-s", str(rate),
        "-p", str(pitch),
        "-a", str(volume),
        "-w", str(out_path),
        text,
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"espeak-ng failed (exit {res.returncode})\nSTDERR:\n{res.stderr.decode(errors='replace')}")

# --------------------------

def main():
    p = argparse.ArgumentParser(description="TTS: EN via Coqui / UA via eSpeak-NG (robust)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", type=str, help="Single sentence")
    src.add_argument("--input_csv", type=str, help="CSV with at least 'text'; optional 'id'")

    p.add_argument("--output_dir", default="tts_out")
    p.add_argument("--filename_prefix", default="utt_")

    # language switch
    p.add_argument("--lang", choices=["en", "uk"], required=True)

    # English (Coqui)
    p.add_argument("--en_model", default="tts_models/en/ljspeech/tacotron2-DDC")

    # Ukrainian (eSpeak-NG)
    p.add_argument("--uk_voice", default="uk", help="eSpeak-NG voice code (default 'uk')")
    p.add_argument("--uk_rate", type=int, default=170)
    p.add_argument("--uk_pitch", type=int, default=50)
    p.add_argument("--uk_volume", type=int, default=200)

    args = p.parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    def synth_one(text: str, out_path: Path):
        if args.lang == "en":
            tts_en_coqui(text, out_path, args.en_model)
        else:
            tts_uk_espeak(text, out_path, voice=args.uk_voice, rate=args.uk_rate, pitch=args.uk_pitch, volume=args.uk_volume)

    if args.text is not None:
        out_path = out_dir / f"{args.filename_prefix}single.wav"
        synth_one(args.text, out_path)
        print(f"[DONE] {out_path}")
        return

    # batch mode
    csv_path = Path(args.input_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        if not rdr.fieldnames or "text" not in rdr.fieldnames:
            raise ValueError("CSV must include at least a 'text' column.")
        for i, row in enumerate(rdr, start=1):
            rid = (row.get("id") or f"{i}").strip()
            text = (row.get("text") or "").strip()
            if not text:
                print(f"[WARN] Row {i}: empty text; skipping.")
                continue
            safe_rid = "".join(c for c in rid if c.isalnum() or c in ("-", "_"))
            out_path = out_dir / f"{args.filename_prefix}{safe_rid}.wav"
            synth_one(text, out_path)
            print(f"[OK] {out_path.name}")

if __name__ == "__main__":
    main()