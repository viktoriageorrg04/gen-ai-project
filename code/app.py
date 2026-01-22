"""
Brand Identity Creative Support Tool - Gradio Web Interface

A multi-agent system that combines LLM, Diffusion, and CLIP models
to support consistent brand identity generation.

Usage:
    python app.py

Then open http://127.0.0.1:7860 in your browser.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Lazy-loaded engines (expensive to initialize)
_identity_engine = None
_visual_generator = None
_quality_critic = None


def get_identity_engine():
    """Lazy-load the Identity Definer (System A)."""
    global _identity_engine
    if _identity_engine is None:
        from system_engines.identity_engine import IdentityDefiner
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        _identity_engine = IdentityDefiner(ollama_url=ollama_url)
    return _identity_engine


def get_visual_generator():
    """Lazy-load the Visual Generator (System B)."""
    global _visual_generator
    if _visual_generator is None:
        from system_engines.gen_engine import VisualGenerator
        _visual_generator = VisualGenerator(backend="diffusers")
    return _visual_generator


def get_quality_critic():
    """Lazy-load the Quality Critic (System C)."""
    global _quality_critic
    if _quality_critic is None:
        from system_engines.critic_engine import QualityCritic
        _quality_critic = QualityCritic()
    return _quality_critic


# ============================================================================
# Handler Functions
# ============================================================================

def generate_manifest(
    concept: str,
    must_include: str,
    avoid: str,
    style_bias: str,
    palette_bias: str,
) -> Dict:
    """
    System A: Generate an identity manifest from a brand concept.
    
    Returns:
        The manifest dict (or empty dict on error)
    """
    # Handle None values from Gradio (empty textboxes return None)
    concept = concept or ""
    must_include = must_include or ""
    avoid = avoid or ""
    style_bias = style_bias or ""
    palette_bias = palette_bias or ""
    
    if not concept.strip():
        gr.Warning("⚠️ Please enter a brand concept.")
        return {}
    
    try:
        engine = get_identity_engine()
    except Exception as e:
        gr.Warning(f"❌ Failed to connect to Ollama. Make sure it's running: ollama serve")
        return {}
    
    # Build constraints from UI inputs
    constraints = {}
    if must_include.strip():
        constraints["must_include"] = [t.strip() for t in must_include.split(",") if t.strip()]
    if avoid.strip():
        constraints["avoid"] = [t.strip() for t in avoid.split(",") if t.strip()]
    if style_bias.strip():
        constraints["style_bias"] = [t.strip() for t in style_bias.split(",") if t.strip()]
    if palette_bias.strip():
        constraints["palette_bias"] = [t.strip() for t in palette_bias.split(",") if t.strip()]
    
    try:
        manifest = engine.generate_manifest(
            user_input=concept,
            constraints=constraints if constraints else None
        )
        gr.Info("✅ Manifest generated successfully!")
        return manifest
    except Exception as e:
        gr.Warning(f"❌ Error generating manifest: {str(e)}")
        return {}


def generate_images(
    manifest_json: Dict,
    scene: str,
    resolution: str,
    num_images: int,
    steps: int,
    guidance_scale: float,
    seed: int,
) -> List[str]:
    """
    System B: Generate images from manifest + scene description.
    
    Returns:
        List of image paths (or empty list on error)
    """
    if not manifest_json:
        gr.Warning("⚠️ Please generate a manifest first (Step 1).")
        return []
    
    # Handle None from Gradio
    scene = scene or ""
    if not scene.strip():
        gr.Warning("⚠️ Please enter a scene description.")
        return []
    
    # Parse resolution string (e.g., "512x512" -> width=512, height=512)
    try:
        width, height = map(int, resolution.split("x"))
    except:
        width, height = 512, 512  # Safe default
    
    try:
        engine = get_identity_engine()
        generator = get_visual_generator()
    except Exception as e:
        gr.Warning(f"❌ Failed to load models: {str(e)}")
        return []
    
    # Create scene prompt from manifest
    try:
        prompt_data = engine.create_scene_prompt(
            manifest=manifest_json,
            scenario=scene,
            include_negative=True
        )
    except Exception as e:
        gr.Warning(f"❌ Error creating prompt: {str(e)}")
        return []
    
    # Determine output directory
    output_dir = Path(os.getenv("OUTPUT_DIR", "outputs")) / "images" / "gradio_session"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle seed (-1 means random)
    actual_seed = None if seed < 0 else seed
    
    try:
        image_paths = generator.generate(
            prompt=prompt_data["prompt"],
            negative_prompt=prompt_data.get("negative_prompt"),
            num_images=int(num_images),
            width=width,
            height=height,
            steps=int(steps),
            guidance_scale=float(guidance_scale),
            seed=actual_seed,
            output_dir=str(output_dir),
        )
        gr.Info(f"✅ Generated {len(image_paths)} image(s)!")
        return image_paths
    except Exception as e:
        gr.Warning(f"❌ Error generating images: {str(e)}")
        return []


def score_images(
    manifest_json: Dict,
    gallery_data: List
) -> Dict:
    """
    System C: Score images against the identity manifest.
    
    Returns:
        Scores dict (or empty dict on error)
    """
    if not manifest_json:
        gr.Warning("⚠️ Please generate a manifest first (Step 1).")
        return {}
    
    if not gallery_data:
        gr.Warning("⚠️ Please generate images first (Step 2).")
        return {}
    
    # Extract image paths from gallery data
    # Gradio gallery returns list of (filepath, caption) tuples or just filepaths
    image_paths = []
    for item in gallery_data:
        if isinstance(item, tuple):
            image_paths.append(item[0])
        elif isinstance(item, dict) and "name" in item:
            image_paths.append(item["name"])
        else:
            image_paths.append(str(item))
    
    if not image_paths:
        gr.Warning("⚠️ No valid images found to score.")
        return {}
    
    try:
        critic = get_quality_critic()
    except Exception as e:
        gr.Warning(f"❌ Failed to load CLIP model: {str(e)}")
        return {}
    
    try:
        all_scores = critic.score_images(image_paths, manifest_json)
        
        # Format for display
        formatted_scores = {}
        for i, score_data in enumerate(all_scores):
            img_name = Path(score_data["image_path"]).name
            formatted_scores[f"Image {i+1} ({img_name})"] = {
                "Overall Score": round(score_data["scores"]["final_score"], 3),
                "Prompt Similarity": round(score_data["scores"]["prompt_similarity"], 3),
                "Feature Alignment": round(score_data["scores"]["feature_similarity"], 3) if score_data["scores"]["feature_similarity"] else "N/A",
                "Style Match": round(score_data["scores"]["style_similarity"], 3) if score_data["scores"]["style_similarity"] else "N/A",
                "Palette Match": round(score_data["scores"]["palette_similarity"], 3) if score_data["scores"]["palette_similarity"] else "N/A",
            }
        
        gr.Info("✅ Scoring complete!")
        return formatted_scores
    except Exception as e:
        gr.Warning(f"❌ Error scoring images: {str(e)}")
        return {}


def refine_manifest_from_scores(
    manifest_json: Dict,
    gallery_data: List,
) -> Tuple[Dict, str]:
    """
    Use critic feedback to refine the manifest via LLM.
    
    Returns:
        Tuple of (refined_manifest, feedback_text)
    """
    if not manifest_json:
        gr.Warning("⚠️ Please generate a manifest first (Step 1).")
        return {}, ""
    
    if not gallery_data:
        gr.Warning("⚠️ Please generate and score images first (Steps 2-3).")
        return {}, ""
    
    image_paths = []
    for item in gallery_data:
        if isinstance(item, tuple):
            image_paths.append(item[0])
        elif isinstance(item, dict) and "name" in item:
            image_paths.append(item["name"])
        else:
            image_paths.append(str(item))
    
    if not image_paths:
        gr.Warning("⚠️ No valid images found.")
        return {}, ""
    
    try:
        critic = get_quality_critic()
        engine = get_identity_engine()
    except Exception as e:
        gr.Warning(f"❌ Failed to load models: {str(e)}")
        return {}, ""
    
    try:
        score_data = critic.score_image(image_paths[0], manifest_json)
        feedback = critic.generate_feedback(score_data, manifest_json)
        refined = engine.refine_manifest(manifest_json, feedback)
        gr.Info("✅ Manifest refined based on critic feedback!")
        return refined, feedback
    except Exception as e:
        gr.Warning(f"❌ Error refining manifest: {str(e)}")
        return {}, ""


# ============================================================================
# Gradio UI Layout
# ============================================================================

def create_app() -> gr.Blocks:
    """Create and return the Gradio application."""
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="Brand Identity Tool",
        css="""
            .step-header { font-size: 1.2em; font-weight: bold; margin-bottom: 0.5em; }
            .status-box { padding: 10px; border-radius: 5px; margin-top: 10px; }
        """
    ) as app:
        
        # Header
        gr.Markdown("""
        # 🎨 Brand Identity Creative Support Tool
        
        Generate **consistent visuals** for your brand character, mascot, logo subject, or product design.
        
        **How it works (Multi-Agent Loop):**
        1. **Describe** a visual subject → LLM creates identity manifest
        2. **Generate** images of that subject in scenes → Diffusion creates visuals
        3. **Score** alignment against the manifest → CLIP evaluates consistency
        4. **Refine** the manifest based on scores → LLM improves, loop back to Step 2
        """)
        
        # Shared state for manifest and images
        manifest_state = gr.State({})
        
        # ====================================================================
        # STEP 1: Define Identity
        # ====================================================================
        with gr.Accordion("📋 Step 1: Define Your Visual Subject", open=True):
            gr.Markdown("*Describe a character, mascot, or visual entity. The AI will create a detailed 'identity manifest' defining its features, colors, and style.*")
            
            concept_input = gr.Textbox(
                label="Visual Subject",
                placeholder="e.g., A futuristic samurai cat with glowing blue eyes",
                lines=2,
                info="Describe what you want to draw: a character, mascot, logo subject, or product.",
            )
            
            with gr.Row():
                gr.Examples(
                    examples=[
                        ["A robotic owl with brass gears and glowing amber eyes"],
                        ["A sleek electric sports car with cyan LED accents"],
                        ["A friendly coffee cup mascot with steam swirls"],
                    ],
                    inputs=concept_input,
                    label="Example Subjects",
                )
            
            with gr.Accordion("🎛️ Advanced Constraints (optional)", open=False):
                gr.Markdown("*Steer the manifest generation with constraints:*")
                
                with gr.Row():
                    must_include_input = gr.Textbox(
                        label="Must Include (comma-separated)",
                        placeholder="e.g., glowing eyes, metallic armor",
                        lines=1,
                    )
                    avoid_input = gr.Textbox(
                        label="Avoid (comma-separated)",
                        placeholder="e.g., cartoonish, low quality",
                        lines=1,
                    )
                
                with gr.Row():
                    style_bias_input = gr.Textbox(
                        label="Style Bias (comma-separated)",
                        placeholder="e.g., Octane Render, cinematic lighting",
                        lines=1,
                    )
                    palette_bias_input = gr.Textbox(
                        label="Palette Bias (comma-separated)",
                        placeholder="e.g., #FF6B00, deep purple, neon cyan",
                        lines=1,
                    )
            
            manifest_btn = gr.Button("🚀 Generate Manifest", variant="primary")
            manifest_output = gr.JSON(label="Identity Manifest")
        
        # ====================================================================
        # STEP 2: Generate Visuals
        # ====================================================================
        with gr.Accordion("🖼️ Step 2: Place Subject in a Scene", open=True):
            gr.Markdown("*Your subject will be placed in this scene. The manifest ensures consistent appearance across different scenes.*")
            
            scene_input = gr.Textbox(
                label="Scene Description",
                placeholder="e.g., perched on a cherry blossom branch at sunset",
                lines=2,
                info="Where should your subject appear? Describe the setting, lighting, and mood.",
            )
            
            with gr.Row():
                gr.Examples(
                    examples=[
                        ["standing in a neon-lit Tokyo alley at night"],
                        ["displayed on a minimalist white product backdrop"],
                        ["posed heroically on a cliff at golden hour"],
                    ],
                    inputs=scene_input,
                    label="Example Scenes",
                )
            
            with gr.Row():
                resolution_input = gr.Dropdown(
                    label="Resolution",
                    choices=["512x512", "768x768", "1024x1024"],
                    value="512x512",  # Safe default for 6GB GPUs
                    info="Lower = faster, less VRAM. 512x512 recommended for <8GB GPU"
                )
                num_images_input = gr.Slider(
                    label="Number of Images",
                    minimum=1,
                    maximum=4,
                    value=1,
                    step=1,
                )
                steps_input = gr.Slider(
                    label="Inference Steps",
                    minimum=10,
                    maximum=50,
                    value=30,
                    step=5,
                )
            
            with gr.Row():
                guidance_input = gr.Slider(
                    label="Guidance Scale",
                    minimum=1.0,
                    maximum=15.0,
                    value=7.0,
                    step=0.5,
                )
                seed_input = gr.Number(
                    label="Seed (-1 = random)",
                    value=-1,
                    precision=0,
                )
            
            generate_btn = gr.Button("🖼️ Generate Images", variant="primary")
            gallery_output = gr.Gallery(
                label="Generated Images (click to view full size)",
                columns=2,  # Fewer columns = larger thumbnails
                rows=1,
                height=400,  # Fixed height so it doesn't overflow
                object_fit="scale-down",  # Scale down to fit, don't crop
                preview=True,  # Allow clicking to see full size
            )
        
        # ====================================================================
        # STEP 3: Evaluate Alignment
        # ====================================================================
        with gr.Accordion("📊 Step 3: Evaluate Alignment", open=True):
            gr.Markdown("*Score how well the generated images match the identity manifest.*")
            
            score_btn = gr.Button("📊 Score Images", variant="primary")
            scores_output = gr.JSON(label="Alignment Scores")
        
        # ====================================================================
        # STEP 4: Refine Based on Feedback
        # ====================================================================
        with gr.Accordion("🔄 Step 4: Refine Manifest (Agent Loop)", open=True):
            gr.Markdown("*The critic's feedback is used to improve the manifest. This closes the multi-agent loop.*")
            
            refine_btn = gr.Button("🔄 Refine Manifest Based on Scores", variant="primary")
            feedback_output = gr.Textbox(
                label="Critic Feedback (sent to LLM)",
                lines=3,
                interactive=False,
            )
            refined_manifest_output = gr.JSON(label="Refined Manifest")
            
            with gr.Row():
                apply_btn = gr.Button("✅ Apply Refined Manifest", variant="secondary")
                gr.Markdown("*Applies the refined manifest to Step 1, enabling re-generation with improvements.*")
        
        # ====================================================================
        # Event Handlers
        # ====================================================================
        
        # Step 1: Generate Manifest
        manifest_btn.click(
            fn=generate_manifest,
            inputs=[
                concept_input,
                must_include_input,
                avoid_input,
                style_bias_input,
                palette_bias_input,
            ],
            outputs=[manifest_output],
        ).then(
            fn=lambda m: m,
            inputs=[manifest_output],
            outputs=[manifest_state],
        )
        
        # Step 2: Generate Images
        generate_btn.click(
            fn=generate_images,
            inputs=[
                manifest_state,
                scene_input,
                resolution_input,
                num_images_input,
                steps_input,
                guidance_input,
                seed_input,
            ],
            outputs=[gallery_output],
        )
        
        # Step 3: Score Images
        score_btn.click(
            fn=score_images,
            inputs=[manifest_state, gallery_output],
            outputs=[scores_output],
        )
        
        # Step 4: Refine Manifest
        refine_btn.click(
            fn=refine_manifest_from_scores,
            inputs=[manifest_state, gallery_output],
            outputs=[refined_manifest_output, feedback_output],
        )
        
        # Step 4b: Apply Refined Manifest
        apply_btn.click(
            fn=lambda m: (m, m) if m else ({}, {}),
            inputs=[refined_manifest_output],
            outputs=[manifest_output, manifest_state],
        )
        
        # ====================================================================
        # Footer
        # ====================================================================
        gr.Markdown("""
        ---
        **Multi-Agent Architecture:** 
        System A (LLM) → System B (Diffusion) → System C (CLIP) → **Loop back to System A**
        
        *Generative AI Research Project - Brand Identity Creative Support Tool*
        """)
    
    return app


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("Starting Brand Identity Creative Support Tool...")
    print("Make sure Ollama is running: ollama serve")
    print()
    
    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
