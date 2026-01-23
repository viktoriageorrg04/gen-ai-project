import os
from typing import List, Optional

try:
    import torch
except ImportError:
    torch = None

try:
    from diffusers import StableDiffusionXLPipeline
except ImportError:
    StableDiffusionXLPipeline = None


class SDXLGenerator:
    """
    SDXL-only generator for side-by-side comparisons.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        dtype: str = "float16",
    ) -> None:
        if StableDiffusionXLPipeline is None or torch is None:
            raise ImportError(
                "diffusers/torch not installed. Install with: pip install diffusers transformers accelerate torch safetensors"
            )

        self.model_id = model_id or os.getenv("DIFFUSERS_SDXL_MODEL_ID") or "stabilityai/stable-diffusion-xl-base-1.0"
        self.device = device or os.getenv("DIFFUSERS_DEVICE") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype or os.getenv("DIFFUSERS_DTYPE") or "float16"

        torch_dtype = torch.float16 if self.dtype == "float16" and self.device == "cuda" else torch.float32

        if os.getenv("DIFFUSERS_DEBUG", "0") == "1":
            print(
                f"[sdxl] model_id={self.model_id} device={self.device} dtype={torch_dtype}"
            )
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"[torch] gpu={name} vram_gb={total:.2f}")

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
        )
        self.pipe.to(self.device)

        if os.getenv("DIFFUSERS_ATTENTION_SLICING", "0") == "1":
            self.pipe.enable_attention_slicing()
        if os.getenv("DIFFUSERS_VAE_SLICING", "0") == "1":
            self.pipe.enable_vae_slicing()
        if os.getenv("DIFFUSERS_CPU_OFFLOAD", "0") == "1":
            self.pipe.enable_model_cpu_offload()

    def _apply_loras(self, lora_paths: Optional[List]) -> None:
        if not lora_paths:
            return
        for item in lora_paths:
            if isinstance(item, dict):
                path = item.get("path")
                scale = item.get("scale", 0.8)
            else:
                path = item
                scale = 0.8
            if not path:
                continue
            adapter_name = os.path.splitext(os.path.basename(str(path)))[0]
            self.pipe.load_lora_weights(path, adapter_name=adapter_name)
            if hasattr(self.pipe, "set_adapters"):
                self.pipe.set_adapters([adapter_name], adapter_weights=[scale])
            elif hasattr(self.pipe, "fuse_lora"):
                self.pipe.fuse_lora(lora_scale=scale)

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
        lora_paths: Optional[List] = None,
        output_dir: str = "outputs/images/sdxl",
    ) -> List[str]:
        os.makedirs(output_dir, exist_ok=True)

        self._apply_loras(lora_paths)

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
            path = os.path.join(output_dir, f"sdxl_{i + 1}.png")
            image.save(path)
            paths.append(path)
        return paths
