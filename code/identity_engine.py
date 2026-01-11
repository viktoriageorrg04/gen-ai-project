import json
import os
from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError:
    requests = None


class IdentityDefiner:
    def __init__(self, model_name=None, ollama_url="http://localhost:11434"):
        """
        Initializes the Ollama connection.
        """
        self.ollama_url = ollama_url
        self.model_name = model_name or "llama3"

        if requests is None:
            raise ImportError("requests package not installed. Run: pip install requests")

        # test connection
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"Ollama server not accessible at {self.ollama_url}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.ollama_url}. Ensure Ollama is running. Error: {e}"
            )

    def generate_manifest(self, user_input: str, constraints: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Takes a raw user concept and returns a structured JSON Identity Manifest.
        Optional constraints allow users to steer or lock parts of the output.
        """
        # prompt is diffusion-aware, meaning that it asks for "art_style_tokens", which are more effective for the vis generator (System B)
        system_prompt = """
### ROLE
You are a Brand Identity Specialist and Prompt Engineer for Generative AI.

### GOAL
Transform a vague user brand concept into a 'Technical Visual Manifest'. This manifest will be used as a 'Master Template' to ensure every generated asset remains 100% consistent in a "Multi-Agent" system.

### INPUT CONCEPT
{user_concept}

### OUTPUT REQUIREMENTS
Return ONLY a valid JSON object and do not include markdown code blocks (```json).
Focus on invariant descriptors: traits that remain unchanged regardless of the character's pose or environment.

### SCHEMA
{
  "core_subject": "A concise, technical description of the main entity.",
  "key_features": ["3-5 permanent, distinctive physical tokens (e.g., 'signature orange goggles', 'glowing cyan circuit-patterns on chest')"],
  "color_palette": ["3-5 specific hex codes or artist-grade color names (e.g., 'International Klein Blue')"],
  "art_style_tokens": ["3-5 technical style keywords (e.g., 'Octane Render', 'Subsurface Scattering', '4k macro photography')"],
  "brand_vibe": "A 2-word emotional anchor (e.g., 'Industrial Solitude')",
  "fixed_negative_prompt": "Tokens to prevent visual drift (e.g., 'low resolution, cartoonish, inconsistent colors, human-features')"
}

### CONSTRAINTS
- Avoid abstract words like "beautiful" or "cool".
- Use "Visual Tokens": words that have a strong, measurable effect on Diffusion models.
- Ensure the 'art_style_tokens' are compatible with the 'core_subject'.
"""

        # this bit enables users to inject constraints into the prompt and steer processing as preferred
        constraints_text = ""
        if constraints:
            must_include = constraints.get("must_include", [])
            avoid = constraints.get("avoid", [])
            style_bias = constraints.get("style_bias", [])
            palette_bias = constraints.get("palette_bias", [])
            locked_fields = constraints.get("locked_fields", {})
            extra_instructions = constraints.get("extra_instructions", "")

            if any([must_include, avoid, style_bias, palette_bias, locked_fields, extra_instructions]):
                lines = ["\n### USER CONSTRAINTS"]
                if must_include:
                    lines.append(f"- Must include: {', '.join(must_include)}")
                if avoid:
                    lines.append(f"- Avoid: {', '.join(avoid)}")
                if style_bias:
                    lines.append(f"- Style bias: {', '.join(style_bias)}")
                if palette_bias:
                    lines.append(f"- Palette bias: {', '.join(palette_bias)}")
                if locked_fields:
                    locked_json = json.dumps(locked_fields, ensure_ascii=True)
                    lines.append(f"- Locked fields (use exact values): {locked_json}")
                if extra_instructions:
                    lines.append(f"- Extra: {extra_instructions}")
                constraints_text = "\n".join(lines)

        formatted_system_prompt = system_prompt.replace("{user_concept}", user_input) + constraints_text
        user_message = f"Create a visual identity manifest for: {user_input}"

        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": formatted_system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    "stream": False,
                    "format": "json",
                },
                timeout=120,
            )
            response.raise_for_status()
            manifest_text = response.json()["message"]["content"]

            # parse and validate the JSON response
            manifest = json.loads(manifest_text)

            # validate required keys
            required_keys = [
                "core_subject",
                "key_features",
                "color_palette",
                "art_style_tokens",
                "brand_vibe",
                "fixed_negative_prompt",
            ]
            missing_keys = [key for key in required_keys if key not in manifest]
            if missing_keys:
                raise ValueError(f"LLM response missing required keys: {missing_keys}")

            return manifest

        except json.JSONDecodeError as e:
            raise ValueError(f"LLM did not return valid JSON. Response: {manifest_text[:200]}...") from e
        except Exception as e:
            raise RuntimeError(f"Error generating manifest with Ollama: {str(e)}") from e

    def create_scene_prompt(
        self,
        manifest: Dict,
        scenario: str,
        include_negative: bool = True,
        extra_tokens: Optional[List[str]] = None,
        extra_negative: Optional[List[str]] = None,
        override_style_tokens: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Combines the Identity Manifest with a specific user scenario to create the final prompt for System B.
        """
        # join the list of tokens into a string for the diffusion model
        features = ", ".join(manifest.get("key_features", []))
        colors = ", ".join(manifest.get("color_palette", []))
        style_tokens = override_style_tokens or manifest.get("art_style_tokens", [])
        style = ", ".join(style_tokens)
        brand_vibe = manifest.get("brand_vibe", "")
        extra = ", ".join(extra_tokens or [])
        prompt = (
            f"{style} portrait of {manifest['core_subject']}, "
            f"{features}, color scheme: {colors}. "
            f"Brand vibe: {brand_vibe}. Scene: {scenario}. "
            f"Additional constraints: {extra}. High resolution, consistent lighting, professional quality."
        )

        result = {"prompt": prompt}
        if include_negative:
            negative = manifest.get("fixed_negative_prompt", "")
            if isinstance(negative, list):
                negative = ", ".join([str(item) for item in negative if str(item).strip()])
            if extra_negative:
                negative = f"{negative}, {', '.join(extra_negative)}" if negative else ", ".join(extra_negative)
            result["negative_prompt"] = negative

        return result

    def save_manifest(self, manifest: Dict, filepath: str) -> None:
        """
        Saves an Identity Manifest to a JSON file.
        """
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Manifest saved to {filepath}")

    def load_manifest(self, filepath: str) -> Dict:
        """
        Loads an Identity Manifest from a JSON file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Manifest file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        print(f"Manifest loaded from {filepath}")
        return manifest


# run through the Terminal
def _basic_ollama_smoke_test() -> None:
    definer = IdentityDefiner(model_name="llama3")
    user_input = "A futuristic samurai cat with neon whiskers"
    manifest = definer.generate_manifest(user_input)

    required_keys = [
        "core_subject",
        "key_features",
        "color_palette",
        "art_style_tokens",
        "brand_vibe",
        "fixed_negative_prompt",
    ]
    missing = [key for key in required_keys if key not in manifest]
    if missing:
        raise ValueError(f"Manifest missing required keys: {missing}")

    scene = "sitting in a ramen shop in Tokyo"
    final_prompt = definer.create_scene_prompt(manifest, scene)
    if not final_prompt.get("prompt"):
        raise ValueError("Scene prompt was empty")
    print("Final prompt:", final_prompt["prompt"][:100] + "...")
    print(
    "[SUCCESS] Smoke test passed.")


if __name__ == "__main__":
    try:
        print("--- TESTING SYSTEM A (OLLAMA) ---")
        _basic_ollama_smoke_test()
    except Exception as e:
        print(f"\n[ERROR]: {e}")
