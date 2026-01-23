# GenAI Brand Identity Tool

A multi-agent workflow for generating **consistent brand visuals**. The pipeline turns a short concept into a structured identity manifest (System A), generates images with diffusion (System B), and scores alignment with CLIP (System C).

## What this repo contains
- **Notebook workflow**: `code/main_interface.ipynb` (Systems A/B/C in sequence, plus experiments).
- **Gradio app**: `code/app.py` (interactive UI with manifest -> generate -> score -> refine loop).
- **Core engines**:
  - `code/system_engines/identity_engine.py` (System A, Ollama LLM)
  - `code/system_engines/gen_engine.py` (System B, Diffusers)
  - `code/system_engines/critic_engine.py` (System C, CLIP)
  - `code/system_engines/gen_engine_v2.py` (SDXL-only generator)
- **Helpers**:
  - `code/helpers/constraints_ui.py` (ipywidgets constraint UI)
  - `code/helpers/constraint_metrics.py` (constraint scoring + plotting)
  - `code/helpers/env_bootstrap.py` (creates `.env`/`.env.example` defaults)

## Quickstart
### 1) Create a virtual environment (inside `code/`)
```powershell
cd code
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies
```powershell
pip install -r requirements.txt
```

Notes:
- `requirements.txt` pins CUDA-enabled PyTorch (`torch==2.5.1+cu121`). If you are on CPU-only, install a CPU build of PyTorch and adjust the requirements accordingly.
- You can use `requirements_min.txt` if you want the minimal set (it is currently equivalent to full requirements).

### 3) Configure environment
```powershell
copy .env.example .env
```
Edit `code/.env` as needed.

### 4) Run the notebook
Open `code/main_interface.ipynb` and run the cells top-to-bottom in the order shown in the notebook.

### 5) Or run the Gradio app
```powershell
python app.py
```
Then open `http://127.0.0.1:7860` in your browser.

## Configuration (env vars)
Defaults live in `code/.env.example` and `code/helpers/env_bootstrap.py`.

Common settings:
- `OLLAMA_URL` (System A LLM endpoint, default `http://localhost:11434`)
- `DIFFUSERS_MODEL_ID` (System B model id, default `runwayml/stable-diffusion-v1-5`)
- `DIFFUSERS_DEVICE` (`cuda` or `cpu`)
- `DIFFUSERS_DTYPE` (`float16` or `float32`)
- `DIFFUSERS_ATTENTION_SLICING`, `DIFFUSERS_VAE_SLICING`, `DIFFUSERS_CPU_OFFLOAD`, `DIFFUSERS_DEBUG`
- `CONTROLNET_MODEL_ID` (optional, for ControlNet Canny)
- `OUTPUT_DIR` (base output path)

## Model defaults (from code)
- **System A (LLM)**: `llama3` via Ollama (`identity_engine.py`)
- **System B (Diffusion)**: `runwayml/stable-diffusion-v1-5` (`gen_engine.py`)
- **System C (CLIP)**: `openai/clip-vit-base-patch32` (`critic_engine.py`)

## Outputs
- Gradio sessions write images to `outputs/images/gradio_session/` by default.
- Other generation paths default to `outputs/images/` (configurable via `OUTPUT_DIR`).

## Troubleshooting
- **Ollama connection error**: Start the server with `ollama serve`, then pull a model (e.g. `ollama pull llama3`).
- **CUDA OOM**: set `DIFFUSERS_DEVICE=cpu`, reduce resolution, or enable slicing/offload.
- **Missing deps**: reinstall with `pip install -r requirements.txt --force-reinstall` and ensure the venv is activated.
