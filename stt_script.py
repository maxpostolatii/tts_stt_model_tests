#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speech-to-Text (STT) with Whisper

Features:
- Single .wav (--audio_file)
- Folder of .wav (--audio_dir) -> CSV (--output_csv)

Install:
  python3 -m pip install --upgrade pip
  pip install openai-whisper torch pandas soundfile
"""

import argparse
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

def list_wavs(audio_dir: Path) -> List[Path]:
    return sorted(p for p in audio_dir.rglob("*.wav") if p.is_file())

def load_model(model_name: str, device: str):
    import os, sys
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import whisper, torch
    if device == "mps":
        try:
            model = whisper.load_model(model_name, device="mps")
            model.fp16 = False
            return model
        except NotImplementedError as e:
            print(f"[WARN] MPS unsupported op ({e}); falling back to CPU.", file=sys.stderr)
            device = "cpu"
    model = whisper.load_model(model_name, device=device)
    if device in ("cpu", "mps"):
        model.fp16 = False
    return model

def transcribe_one(model, wav_path: Path, language: Optional[str]) -> Dict:
    result = model.transcribe(str(wav_path), language=language, fp16=getattr(model, "fp16", False))
    return {
        "filename": wav_path.name,
        "path": str(wav_path),
        "language": result.get("language"),
        "text": result.get("text", "").strip()
    }

def main():
    ap = argparse.ArgumentParser(description="Transcribe WAV(s) with Whisper")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--audio_file", type=str, help="Path to a single .wav file.")
    src.add_argument("--audio_dir", type=str, help="Path to a directory with .wav files.")

    ap.add_argument("--output_csv", type=str, help="Where to save CSV.")
    ap.add_argument("--model", default="small", help="Whisper model size: tiny|base|small|medium|large")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"], help="Device (M1: use mps).")
    ap.add_argument("--language", default=None, help="Language code (en, uk) or None for auto-detect.")
    args = ap.parse_args()

    # Collect files
    if args.audio_file:
        files = [Path(args.audio_file)]
    else:
        files = list_wavs(Path(args.audio_dir))
        if not files:
            ap.error("No .wav files found")

    # Load model
    print(f"[INFO] Loading Whisper '{args.model}' on {args.device} ...")
    model = load_model(args.model, args.device)

    rows: List[Dict] = []
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.name}")
        try:
            row = transcribe_one(model, f, args.language)
        except Exception as e:
            row = {"filename": f.name, "path": str(f), "language": None, "text": "", "error": str(e)}
        rows.append(row)

    df = pd.DataFrame(rows)
    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8")
        print(f"[DONE] Wrote: {out}")
    else:
        print(df.to_csv(index=False))

if __name__ == "__main__":
    main()