# TTS & STT Homework Project

This project demonstrates **Text-to-Speech (TTS)** and **Speech-to-Text (STT)** pipelines using Python.  
It supports both **English** and **Ukrainian** text/audio processing.

## Features

- **TTS (Text-to-Speech)**
  - English: Coqui TTS (Tacotron2-DDC model).
  - Ukrainian: Silero TTS (alternative to Piper).
  - Supports both single text input and batch synthesis from CSV.

- **STT (Speech-to-Text)**
  - Based on OpenAI Whisper.
  - Supports transcription from individual audio files or batch processing from CSV.

## Installation

Make sure you have **Python 3.11+** installed.  
Then install dependencies:

```bash
pip3 install torch soundfile numpy TTS openai-whisper
```

If you are on macOS and encounter `MPS` backend issues with Whisper, force CPU mode using:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Usage

### TTS (Text-to-Speech)

#### 1. English — single text
```bash
python3.11 tts_script.py --lang en --text "Hello! This is a TTS demo."
```

#### 2. English — batch from CSV
```bash
python3.11 tts_script.py --lang en --input_csv input_en.csv
```

#### 3. Ukrainian — single text
```bash
python3.11 tts_script.py --lang uk --text "Привіт! Це демонстрація синтезу."
```

#### 4. Ukrainian — batch from CSV
```bash
python3.11 tts_script.py --lang uk --input_csv input_uk.csv
```

### Example CSVs

**`input_en.csv`**
```csv
id,text
1,Hello! This is a demo.
2,We are testing batch text-to-speech synthesis.
```

**`input_uk.csv`**
```csv
id,text
1,Привіт! Це демонстрація синтезу.
2,Ми тестуємо пакетний синтез мовлення з тексту.
```

### STT (Speech-to-Text)

#### 1. Single audio file
```bash
python3.11 stt_script.py --audio_file tts_out/utt_single.wav
```

#### 2. Batch from CSV
```bash
python3.11 stt_script.py --input_csv input_audio.csv
```

**`input_audio.csv`**
```csv
id,audio_file
1,tts_out/utt_single.wav
2,tts_out/utt_2.wav
```

## Notes

- TTS outputs are stored in `tts_out/`.
- STT transcriptions are printed to console and can be extended to save results in CSV.
- For Ukrainian voices, Silero supports multiple speakers (`v4_ua`, `v3_ua`, `mykyta_v2`).  
  You can select a speaker with `--uk_speaker`.

---
