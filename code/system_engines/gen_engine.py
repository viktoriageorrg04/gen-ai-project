import os
from typing import List, Optional

try:
    import torch
except ImportError:
    torch = None

try:
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
except ImportError:
    StableDiffusionPipeline = None
    StableDiffusionXLPipeline = None

class VisualGenerator:
    """
    System B: Visual Generator (Diffusion + Adapters).
    Supports a local diffusers pipeline.
    """

    def __init__(
        self,
        backend: str = "diffusers",
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        dtype: str = "float16",
    ) -> None:
        self.backend = backend.lower()
        self.model_id = model_id or os.getenv("DIFFUSERS_MODEL_ID")
        self.device = device or os.getenv("DIFFUSERS_DEVICE") or (
            "cuda" if torch and torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype or os.getenv("DIFFUSERS_DTYPE") or "float16"

        self.pipe = None

        if self.backend == "diffusers":
            self._init_diffusers()
        else:
            raise ValueError("Only the 'diffusers' backend is supported.")

    def _init_diffusers(self) -> None:
        if StableDiffusionPipeline is None or StableDiffusionXLPipeline is None or torch is None:
            raise ImportError(
                "diffusers/torch not installed. Install with: pip install diffusers transformers accelerate torch safetensors"
            )

        model_id = self.model_id or "runwayml/stable-diffusion-v1-5"
        use_xl = "xl" in model_id.lower()
        torch_dtype = torch.float16 if self.dtype == "float16" and self.device == "cuda" else torch.float32

        if os.getenv("DIFFUSERS_DEBUG", "0") == "1":
            print(
                f"[diffusers] model_id={model_id} use_xl={use_xl} device={self.device} dtype={torch_dtype}"
            )
            if torch is not None:
                print(f"[torch] cuda_available={torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    name = torch.cuda.get_device_name(0)
                    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    print(f"[torch] gpu={name} vram_gb={total:.2f}")

        try:
            if use_xl:
                self.pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
            else:
                self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and self.device == "cuda":
                raise RuntimeError(
                    "CUDA OOM while loading the model. Try DIFFUSERS_DEVICE=cpu, "
                    "set a smaller model_id, or enable offload/slicing flags."
                ) from exc
            raise

        self.pipe.to(self.device)

        if os.getenv("DIFFUSERS_ATTENTION_SLICING", "0") == "1":
            self.pipe.enable_attention_slicing()
        if os.getenv("DIFFUSERS_VAE_SLICING", "0") == "1":
            self.pipe.enable_vae_slicing()
        if os.getenv("DIFFUSERS_CPU_OFFLOAD", "0") == "1":
            self.pipe.enable_model_cpu_offload()

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_images: int = 1,
        width: int = 1024,
        height: int = 1024,
        steps: int = 30,
        guidance_scale: float = 7.0,
        seed: Optional[int] = None,
        lora_paths: Optional[List[str]] = None,
        ip_adapter: Optional[str] = None,
        output_dir: str = "outputs/images",
    ) -> List[str]:
        """
        Generate images for System B. Returns file paths.
        """
        os.makedirs(output_dir, exist_ok=True)

        if self.backend == "diffusers":
            return self._generate_diffusers(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images=num_images,
                width=width,
                height=height,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                output_dir=output_dir,
            )

        raise ValueError("Only the 'diffusers' backend is supported.")

    def _generate_diffusers(
        self,
        prompt: str,
        negative_prompt: Optional[str],
        num_images: int,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: Optional[int],
        output_dir: str,
    ) -> List[str]:
        if self.pipe is None:
            raise RuntimeError("Diffusers pipeline not initialized.")

        generator = None
        if seed is not None and torch is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        paths = []
        for i, image in enumerate(result.images):
            path = os.path.join(output_dir, f"image_{i + 1}.png")
            image.save(path)
            paths.append(path)
        return paths
