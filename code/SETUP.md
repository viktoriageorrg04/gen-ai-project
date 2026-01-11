# Setup Guide for GenAI Brand Identity Tool

### install dependencies (project-only env)
```powershell
# create a dedicated virtual env
python -m venv .venv

# activate it
.\.venv\Scripts\Activate.ps1

# install packages
pip install -r requirements.txt

# register a notebook kernel for this env
python -m ipykernel install --user --name genai-brand --display-name "GenAI Brand"
```

### config env
```powershell
# copy the example env file
cp .env.example .env

# edit .env and add your settings
```

### test System A (Identity Definer)
```powershell
# using Ollama (local)
# first install Ollama: https://ollama.ai
ollama pull llama3
python identity_engine.py
```

## installation options

### option 1: minimal install (System A only)
```powershell
pip install requests python-dotenv
```

### option 2: full stack (recommended)
```powershell
pip install -r requirements.txt

# for GPU support (CUDA) if you need a specific build:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## dir structure
```
GenAI Project/
├── code/
│   ├── identity_engine.py    # System A (LLM)
│   ├── gen_engine.py          # System B (Diffusion)
│   ├── critic_engine.py       # System C (VLM)
│   ├── requirements.txt       # Dependencies
│   └── .env                   # Your API keys (create this)
├── data/
│   └── manifests/             # Saved brand identities
└── outputs/
    └── images/                # Generated images
```

## troubleshooting

### Ollama connection error
- Start Ollama: `ollama serve`
- Pull model: `ollama pull llama3`
- Test: `curl http://localhost:11434/api/tags`

### import errors
- Ensure venv is activated: `.\venv\Scripts\Activate.ps1`
- Reinstall packages: `pip install -r requirements.txt --force-reinstall`
