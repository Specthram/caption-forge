# Caption Forge

**Caption Forge is a local app to organize your image & video libraries and
caption them end-to-end with AI assistance.** It scores the visual quality of
your media, helps you generate and refine captions, and ships a
training-ready dataset — all on your own GPU, nothing leaves the machine.

React + FastAPI desktop-style UI, served at <http://127.0.0.1:7776>.

## Typical workflow

1. **Add a library** — register a source folder (or drag-and-drop uploads) on
   the **Libraries** tab, then **Index** it (thumbnails, quality scores,
   similarity/semantic embeddings, WD14 auto-tags).
2. **Tag** — create or import tags, organize them into categories, and attach
   them to media where needed (per image, per-library bulk, or auto-tagged).
3. **Build a dataset** — on **Datasets**, hand-pick media with the composer or
   let the **Auto-build Studio** propose a whole set; media are linked, never
   copied.
4. **Caption** — write captions by hand or **generate** them with a local VLM
   (`.safetensors` / `.gguf`); review, ground against the image and score them.
5. **Clean up** — remove watermarks/logos with the AI **Watermark Lab** and
   **crop** images — both non-destructive, source files never modified.
6. **Deploy** — mirror the finished dataset straight into your training folder,
   resized on the way out and only rewriting what changed.

## Key features

- **Quality & diversity** — MUSIQ / TOPIQ / LAION-Aes / Q-Align scoring, a
  DINOv2 diversity map, near-duplicate detection, and a 0-100 dataset
  **Quality report**.
- **Captioning** — batch VLM generation (Qwen2.5-VL, Qwen3-VL, Gemma 3, Gemma 4,
  JoyCaption…), per-media versions with autosave, integrity review, SigLIP 2
  grounding and zero-reference caption scoring.
- **Rule-based review** — a **Review** sub-tab where a judge model (chosen
  independently from the captioner) checks each caption against plain-language
  rules per dataset (deterministic, text-only, or vision). Every proposed fix
  is human-validated in a keyboard-driven wizard — nothing is applied silently.
- **Tagging** — WD14 auto-tagger, colored tags in categories, and
  folder→tag mapping rules on import.
- **Non-destructive editing** — virtual crops and AI watermark removal
  (OWLv2 / YOLO detect, FLUX.2 Klein for erase); the original files are never touched except if you want to override them explicitly.
- **Deploy** — Copy your dataset in your training tool, export it as zip.

Everything is stored in a single SQLite database under `database/`; media
files are referenced in place, never duplicated.

## Prerequisites

- Python 3.12
- Node.js 20+ (for the React front-end; `install.bat` builds it)
- An NVIDIA GPU with CUDA (12.8 recommended; 12.6 and 12.4 also work —
  RTX 50-series requires 12.8)
- Vision-language model files (`.safetensors` and/or `.gguf`) in the bundled
  `models/` folder

## Installation

### Windows

Run `install.bat` once. It checks for Python 3.12, then asks for the CUDA
version (12.8 / 12.6 / 12.4) and whether to add GGUF support
(`llama-cpp-python`, for `.gguf` models). It creates the `venv` and installs
everything. Then run `run.bat` to launch — and for every launch after.

### Linux

```bash
python3.12 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
( cd web && npm ci && npm run build )   # needs Node.js 20+
python -m uvicorn server.main:app --host 127.0.0.1 --port 7776
```
