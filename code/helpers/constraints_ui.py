import json

try:
    import ipywidgets as widgets
    from IPython.display import display

    _widgets_available = True
except ImportError:
    _widgets_available = False


if not _widgets_available:
    print("ipywidgets not installed. Run: pip install ipywidgets")

    def build_constraints():
        return None

    def build_scene_overrides():
        return {"extra_tokens": None, "extra_negative": None, "override_style_tokens": None}

    def get_constraint_ui_state():
        return {"enabled": False, "constraints": None, "overrides": None}

    def render_constraints_ui():
        return None
else:
    use_constraints = widgets.Checkbox(value=False, description="Enable constraints")
    must_include = widgets.Textarea(description="Must include", placeholder="comma-separated")
    avoid = widgets.Textarea(description="Avoid", placeholder="comma-separated")
    style_bias = widgets.Textarea(description="Style bias", placeholder="comma-separated")
    palette_bias = widgets.Textarea(description="Palette bias", placeholder="comma-separated")
    locked_fields = widgets.Textarea(description="Locked fields (JSON)", placeholder='{"core_subject": "..."}')
    extra_instructions = widgets.Textarea(description="Extra instructions", placeholder="optional")
    extra_tokens = widgets.Textarea(description="Scene extra tokens", placeholder="comma-separated")
    extra_negative = widgets.Textarea(description="Scene negative tokens", placeholder="comma-separated")
    override_style_tokens = widgets.Textarea(description="Override style tokens", placeholder="comma-separated")

    load_demo = widgets.Button(description="Load Demo Constraints", button_style="info")

    def _parse_csv(text):
        return [t.strip() for t in text.split(",") if t.strip()]

    def _parse_locked_fields(text):
        if not text.strip():
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Locked fields must be valid JSON.") from exc

    def _load_demo_constraints(_):
        use_constraints.value = True
        must_include.value = "neon whiskers, katana, cybernetic armor"
        avoid.value = "cartoonish, low-res, deformed"
        style_bias.value = "Octane Render, Subsurface Scattering, high-contrast lighting"
        palette_bias.value = "Electric Blue #03A9F4, Neon Pink #FF69B4, Metallic Silver #B1B1B1"
        locked_fields.value = '{"core_subject": "Feline warrior with advanced cybernetic enhancements"}'
        extra_instructions.value = "Keep the subject non-human and helmeted."
        extra_tokens.value = "cinematic rim light, depth of field"
        extra_negative.value = "blurry, wrong anatomy"
        override_style_tokens.value = "Octane Render, Subsurface Scattering"

    load_demo.on_click(_load_demo_constraints)

    def build_constraints():
        if not use_constraints.value:
            return None
        constraints = {}
        if must_include.value.strip():
            constraints["must_include"] = _parse_csv(must_include.value)
        if avoid.value.strip():
            constraints["avoid"] = _parse_csv(avoid.value)
        if style_bias.value.strip():
            constraints["style_bias"] = _parse_csv(style_bias.value)
        if palette_bias.value.strip():
            constraints["palette_bias"] = _parse_csv(palette_bias.value)
        if locked_fields.value.strip():
            constraints["locked_fields"] = _parse_locked_fields(locked_fields.value)
        if extra_instructions.value.strip():
            constraints["extra_instructions"] = extra_instructions.value.strip()
        return constraints or None

    def build_scene_overrides():
        if not use_constraints.value:
            return {"extra_tokens": None, "extra_negative": None, "override_style_tokens": None}
        overrides = {
            "extra_tokens": _parse_csv(extra_tokens.value) if extra_tokens.value.strip() else None,
            "extra_negative": _parse_csv(extra_negative.value) if extra_negative.value.strip() else None,
            "override_style_tokens": _parse_csv(override_style_tokens.value) if override_style_tokens.value.strip() else None,
        }
        return overrides

    def get_constraint_ui_state():
        return {
            "enabled": use_constraints.value,
            "constraints": build_constraints(),
            "overrides": build_scene_overrides(),
        }

    def render_constraints_ui():
        ui = widgets.VBox(
            [
                use_constraints,
                load_demo,
                must_include,
                avoid,
                style_bias,
                palette_bias,
                locked_fields,
                extra_instructions,
                extra_tokens,
                extra_negative,
                override_style_tokens,
            ]
        )
        display(ui)
        return ui
