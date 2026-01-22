import os
from typing import Dict, List, Optional

try:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    torch = None
    Image = None
    CLIPModel = None
    CLIPProcessor = None


class QualityCritic:
    """
    System C: Quality Critic (VLM).
    Uses CLIP to score image alignment against the Identity Manifest.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ) -> None:
        if torch is None or CLIPModel is None or CLIPProcessor is None or Image is None:
            raise ImportError(
                "Missing VLM deps. Install with: pip install transformers torch pillow"
            )

        self.model_id = model_id or os.getenv("CLIP_MODEL_ID") or "openai/clip-vit-base-patch32"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or ("float16" if self.device == "cuda" else "float32")

        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id, use_safetensors=True)
        self.model.to(self.device)
        if self.device == "cuda" and self.dtype == "float16":
            self.model = self.model.to(dtype=torch.float16)
        self.model.eval()

    def _normalize(self, features: "torch.Tensor") -> "torch.Tensor":
        return features / features.norm(p=2, dim=-1, keepdim=True)

    def _encode_texts(self, texts: List[str]) -> "torch.Tensor":
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return self._normalize(text_features)

    def _encode_image(self, image: "Image.Image") -> "torch.Tensor":
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return self._normalize(image_features)

    def _similarities(self, image: "Image.Image", texts: List[str]) -> List[float]:
        image_features = self._encode_image(image)
        text_features = self._encode_texts(texts)
        scores = text_features @ image_features.T
        sims = scores.squeeze(-1).float().cpu().tolist()
        if isinstance(sims, float):
            sims = [sims]
        return sims

    def build_manifest_prompt(self, manifest: Dict) -> str:
        core_subject = manifest.get("core_subject", "")
        key_features = ", ".join(manifest.get("key_features", []) or [])
        color_palette = ", ".join(manifest.get("color_palette", []) or [])
        art_style_tokens = ", ".join(manifest.get("art_style_tokens", []) or [])
        brand_vibe = manifest.get("brand_vibe", "")

        parts = [core_subject]
        if key_features:
            parts.append(f"features: {key_features}")
        if color_palette:
            parts.append(f"colors: {color_palette}")
        if art_style_tokens:
            parts.append(f"style: {art_style_tokens}")
        if brand_vibe:
            parts.append(f"vibe: {brand_vibe}")
        return " | ".join([p for p in parts if p])

    def _score_group(
        self,
        image: "Image.Image",
        prompt_prefix: str,
        items: List[str],
        core_subject: str,
    ) -> Optional[Dict]:
        if not items:
            return None
        texts = []
        for item in items:
            if core_subject:
                texts.append(f"{prompt_prefix} {core_subject} with {item}")
            else:
                texts.append(f"{prompt_prefix} {item}")
        sims = self._similarities(image, texts)
        scores = [(s + 1) / 2 for s in sims]
        return {
            "mean": sum(scores) / len(scores),
            "items": items,
            "scores": scores,
        }

    def score_image(
        self,
        image_path: str,
        manifest: Dict,
        prompt_override: Optional[str] = None,
    ) -> Dict:
        image = Image.open(image_path).convert("RGB")

        prompt = prompt_override or self.build_manifest_prompt(manifest)
        prompt_score = self._similarities(image, [prompt])[0]
        prompt_score = (prompt_score + 1) / 2

        core_subject = manifest.get("core_subject", "")
        key_features = manifest.get("key_features", []) or []
        art_style_tokens = manifest.get("art_style_tokens", []) or []
        color_palette = manifest.get("color_palette", []) or []
        fixed_negative = manifest.get("fixed_negative_prompt", "")

        feature_scores = self._score_group(image, "a portrait of", key_features, core_subject)
        style_scores = self._score_group(image, "in the style of", art_style_tokens, "")
        palette_scores = self._score_group(image, "color palette", color_palette, "")

        negative_score = None
        if isinstance(fixed_negative, list):
            fixed_negative = ", ".join([str(item) for item in fixed_negative if str(item).strip()])
        if fixed_negative:
            negative_score = self._similarities(image, [fixed_negative])[0]
            negative_score = (negative_score + 1) / 2

        weighted = []
        weights = []

        weighted.append(prompt_score)
        weights.append(0.55)

        if feature_scores is not None:
            weighted.append(feature_scores["mean"])
            weights.append(0.2)
        if style_scores is not None:
            weighted.append(style_scores["mean"])
            weights.append(0.15)
        if palette_scores is not None:
            weighted.append(palette_scores["mean"])
            weights.append(0.05)
        if negative_score is not None:
            weighted.append(1 - negative_score)
            weights.append(0.05)

        total_weight = sum(weights) if weights else 1.0
        final_score = sum(w * s for w, s in zip(weights, weighted)) / total_weight

        return {
            "image_path": image_path,
            "prompt": prompt,
            "scores": {
                "prompt_similarity": prompt_score,
                "feature_similarity": feature_scores["mean"] if feature_scores else None,
                "style_similarity": style_scores["mean"] if style_scores else None,
                "palette_similarity": palette_scores["mean"] if palette_scores else None,
                "negative_similarity": negative_score,
                "final_score": final_score,
            },
            "details": {
                "features": feature_scores,
                "styles": style_scores,
                "palette": palette_scores,
            },
        }

    def score_images(
        self,
        image_paths: List[str],
        manifest: Dict,
        prompt_override: Optional[str] = None,
    ) -> List[Dict]:
        return [self.score_image(path, manifest, prompt_override=prompt_override) for path in image_paths]

    def generate_feedback(
        self,
        score_data: Dict,
        manifest: Dict,
        threshold: float = 0.7,
    ) -> str:
        scores = score_data.get("scores", {})
        details = score_data.get("details", {})
        feedback_parts = []

        feature_sim = scores.get("feature_similarity")
        if feature_sim is not None and feature_sim < threshold:
            feature_details = details.get("features", {})
            weak_features = []
            if feature_details and "items" in feature_details and "scores" in feature_details:
                for item, score in zip(feature_details["items"], feature_details["scores"]):
                    if score < threshold:
                        weak_features.append(f"'{item}' (score: {score:.2f})")
            if weak_features:
                feedback_parts.append(f"Weak key_features that need strengthening: {', '.join(weak_features)}. Make these more visually distinctive and specific.")
            else:
                feedback_parts.append(f"Feature alignment is low ({feature_sim:.2f}). Make key_features more specific and visually distinctive.")

        style_sim = scores.get("style_similarity")
        if style_sim is not None and style_sim < threshold:
            style_details = details.get("styles", {})
            weak_styles = []
            if style_details and "items" in style_details and "scores" in style_details:
                for item, score in zip(style_details["items"], style_details["scores"]):
                    if score < threshold:
                        weak_styles.append(f"'{item}' (score: {score:.2f})")
            if weak_styles:
                feedback_parts.append(f"Weak art_style_tokens: {', '.join(weak_styles)}. Use more specific, diffusion-friendly style tokens.")
            else:
                feedback_parts.append(f"Style alignment is low ({style_sim:.2f}). Use more specific rendering/style tokens.")

        palette_sim = scores.get("palette_similarity")
        if palette_sim is not None and palette_sim < threshold:
            feedback_parts.append(f"Color palette alignment is low ({palette_sim:.2f}). Use more specific color names or hex codes that translate well to images.")

        negative_sim = scores.get("negative_similarity")
        if negative_sim is not None and negative_sim > 0.5:
            feedback_parts.append(f"Unwanted elements detected (negative score: {negative_sim:.2f}). Strengthen fixed_negative_prompt to exclude these elements more explicitly.")

        if not feedback_parts:
            final_score = scores.get("final_score", 0)
            if final_score >= 0.8:
                return "The manifest is performing well. Minor refinements: consider adding more specific visual tokens to key_features for even better consistency."
            else:
                return "General improvement needed: make all visual descriptors more specific and technically precise for diffusion models."

        return " ".join(feedback_parts)


def _basic_clip_smoke_test() -> None:
    critic = QualityCritic()
    manifest = {
        "core_subject": "futuristic samurai cat",
        "key_features": ["neon whiskers", "katana sheath", "cybernetic armor"],
        "color_palette": ["#03A9F4", "#FF69B4", "#B1B1B1"],
        "art_style_tokens": ["Octane Render", "Subsurface Scattering"],
        "brand_vibe": "Rugged Elegance",
        "fixed_negative_prompt": "low resolution, blurry, cartoonish",
    }
    test_image = os.getenv("SYSTEM_C_TEST_IMAGE")
    if not test_image:
        print("Set SYSTEM_C_TEST_IMAGE to run the CLIP smoke test.")
        return
    result = critic.score_image(test_image, manifest)
    print("Final score:", round(result["scores"]["final_score"], 3))


if __name__ == "__main__":
    _basic_clip_smoke_test()
    