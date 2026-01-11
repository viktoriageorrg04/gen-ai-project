from pathlib import Path


DEFAULT_TEMPLATE = """# System A (Ollama)
OLLAMA_URL=http://localhost:11434

# System B (Diffusers)
DIFFUSERS_MODEL_ID=runwayml/stable-diffusion-v1-5
DIFFUSERS_DEVICE=cuda
DIFFUSERS_DTYPE=float16
DIFFUSERS_ATTENTION_SLICING=1
DIFFUSERS_VAE_SLICING=1
DIFFUSERS_CPU_OFFLOAD=1
DIFFUSERS_DEBUG=1

# Paths
OUTPUT_DIR=outputs
DATA_DIR=data
"""


def ensure_env_files(cwd: Path) -> tuple[Path, Path]:
    env_example = cwd / ".env.example"
    env_file = cwd / ".env"

    if not env_example.exists():
        env_example.write_text(DEFAULT_TEMPLATE, encoding="utf-8")
        print("Created .env.example")

    if not env_file.exists():
        env_file.write_text(env_example.read_text(encoding="utf-8"), encoding="utf-8")
        print("Created .env from .env.example")

    return env_example, env_file