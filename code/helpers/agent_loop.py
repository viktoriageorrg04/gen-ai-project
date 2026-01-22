import time
from typing import Dict, List, Tuple, Any

class AutoLoop:
    def __init__(self, identity_engine, visual_generator, quality_critic):
        self.identity_engine = identity_engine
        self.visual_generator = visual_generator
        self.quality_critic = quality_critic

    def run_loop(
        self,
        initial_manifest: Dict,
        scene: str,
        target_score: float = 0.85,
        max_iterations: int = 5,
        # Gen params
        num_images: int = 1,
        steps: int = 30,
        guidance_scale: float = 7.0,
        width: int = 512,
        height: int = 512,
        seed: int = -1
    ):
        """
        Runs the autonomous refinement loop.
        Yields (status_message, history_log, current_manifest) per step.
        """
        history = []
        current_manifest = initial_manifest
        
        # Ensure we have valid models
        if not self.identity_engine or not self.visual_generator or not self.quality_critic:
            yield "❌ Error: AI engines not initialized.", history, current_manifest
            return

        print(f"[AutoLoop] Starting loop. Target: {target_score}, Max Iter: {max_iterations}")

        for i in range(max_iterations):
            iteration_log = {
                "iter": i + 1,
                "manifest": current_manifest,
                "logs": []
            }
            
            # --- Step 1: Generate ---
            try:
                # Create prompt
                prompt_data = self.identity_engine.create_scene_prompt(current_manifest, scene)
                iteration_log["prompt"] = prompt_data["prompt"]
                iteration_log["logs"].append(f"Generating (Iter {i+1})...")
                
                # Generate
                actual_seed = None if seed < 0 else seed + i  # Increment seed per iteration to ensure variety
                image_paths = self.visual_generator.generate(
                    prompt=prompt_data["prompt"],
                    negative_prompt=prompt_data.get("negative_prompt"),
                    num_images=num_images,
                    width=width,
                    height=height,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    seed=actual_seed
                )
                
                # Rename images to prevent overwriting and ensure gallery updates
                # (Constraint: VisualGenerator always saves to same filename if output_dir is constant)
                import os
                unique_paths = []
                for p in image_paths:
                    dirname, basename = os.path.split(p)
                    name, ext = os.path.splitext(basename)
                    new_name = f"{name}_iter{i+1}_{int(time.time()*1000)}{ext}"
                    new_path = os.path.join(dirname, new_name)
                    try:
                        os.rename(p, new_path)
                        unique_paths.append(new_path)
                    except OSError as e:
                        print(f"Error renaming file: {e}")
                        unique_paths.append(p)
                
                if unique_paths:
                    image_paths = unique_paths
                
                if not image_paths:
                    iteration_log["logs"].append("❌ Generation failed.")
                    history.append(iteration_log)
                    yield "❌ Generation failed.", history, current_manifest
                    return
                
                # We interpret the "best" image if multiple are generated
                current_image_path = image_paths[0]
                iteration_log["image"] = current_image_path
                
            except Exception as e:
                err = f"❌ Error in Generation step: {str(e)}"
                print(err)
                iteration_log["logs"].append(err)
                history.append(iteration_log)
                yield err, history, current_manifest
                return

            # --- Step 2: Score ---
            try:
                score_data = self.quality_critic.score_image(current_image_path, current_manifest)
                final_score = score_data["scores"]["final_score"]
                iteration_log["score"] = final_score
                iteration_log["score_data"] = score_data
                
                log_msg = (
                    f"ITERATION {i+1}: Score = {final_score:.3f}\n"
                    f"  - Prompt Alignment: {score_data['scores']['prompt_similarity']:.3f}\n"
                    f"  - Feature Match: {score_data['scores'].get('feature_similarity', 0):.3f}\n"
                    f"  - Style Match: {score_data['scores'].get('style_similarity', 0):.3f}"
                )
                iteration_log["logs"].append(log_msg)
                
                # Check success
                if final_score >= target_score:
                    iteration_log["logs"].append("✅ Target score reached! Stopping loop.")
                    history.append(iteration_log)
                    yield "✅ Success: Target score reached.", history, current_manifest
                    return
                
            except Exception as e:
                err = f"❌ Error in Scoring step: {str(e)}"
                print(err)
                iteration_log["logs"].append(err)
                history.append(iteration_log)
                yield err, history, current_manifest
                return

            # Yield progress after scoring (so user sees the image and score)
            history.append(iteration_log)
            # We continue if max iterations not reached, but yield here to update UI
            yield f"Iter {i+1} Complete (Score: {final_score:.2f})", history, current_manifest

            # Stop if this was the last iteration
            if i == max_iterations - 1:
                iteration_log["logs"].append("⚠️ Max iterations reached.")
                yield "⚠️ Max iterations reached.", history, current_manifest
                return

            # --- Step 3: Refine (only if we are continuing) ---
            try:
                feedback = self.quality_critic.generate_feedback(score_data, current_manifest)
                iteration_log["critic_feedback"] = feedback
                iteration_log["logs"].append(f"Critic Feedback: {feedback}")
                
                iteration_log["logs"].append("Refining manifest...")
                
                # Yield update before calling LLM so user knows what's happening
                yield f"Refining Manifest (Iter {i+1})...", history, current_manifest

                new_manifest = self.identity_engine.refine_manifest(current_manifest, feedback)
                
                if new_manifest == current_manifest:
                    iteration_log["logs"].append("⚠️ Manifest did not change. Stopping early.")
                    yield "⚠️ Stalled: Manifest stopped evolving.", history, current_manifest
                    return
                
                current_manifest = new_manifest
                iteration_log["logs"].append("✅ Manifest updated.")
                
            except Exception as e:
                err = f"❌ Error in Refinement step: {str(e)}"
                print(err)
                iteration_log["logs"].append(err)
                yield err, history, current_manifest
                return
        
        yield "Loop finished.", history, current_manifest
