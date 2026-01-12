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
    core_subject = widgets.Text(description="Core subject", placeholder="e.g., corporate AI platform", layout={"width": "420px"})
    must_include = widgets.Textarea(description="Must include", placeholder="comma-separated", layout={"width": "420px"})
    avoid = widgets.Textarea(description="Avoid", placeholder="comma-separated", layout={"width": "420px"})
    style_bias = widgets.Textarea(description="Style bias", placeholder="comma-separated", layout={"width": "420px"})
    palette_bias = widgets.Textarea(description="Palette bias", placeholder="comma-separated", layout={"width": "420px"})
    locked_fields = widgets.Textarea(
        description="Locked fields (JSON)",
        placeholder='{"core_subject": "..."}',
        layout={"width": "420px"},
    )
    extra_instructions = widgets.Textarea(
        description="Extra instructions", placeholder="optional", layout={"width": "420px"}
    )
    extra_tokens = widgets.Textarea(
        description="Scene extra tokens", placeholder="comma-separated", layout={"width": "420px"}
    )
    extra_negative = widgets.Textarea(
        description="Scene negative tokens", placeholder="comma-separated", layout={"width": "420px"}
    )
    override_style_tokens = widgets.Textarea(
        description="Override style tokens", placeholder="comma-separated", layout={"width": "420px"}
    )

    load_demo = widgets.Button(description="Load Demo Constraints", button_style="info", layout={"width": "200px"})

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
        core_subject.value = "Enterprise AI platform identity mark"
        must_include.value = "clean geometric mark, subtle gradient, balanced whitespace, premium feel"
        avoid.value = "neon, grunge, cartoonish, overly playful, busy background"
        style_bias.value = "minimalism, modern corporate, grid-aligned, sharp edges"
        palette_bias.value = "Deep Navy #0A1F44, Slate #4C5A67, Cool Gray #D9DEE3, Accent Teal #2FB7A6"
        locked_fields.value = '{"brand_vibe": "Trusted Precision"}'
        extra_instructions.value = "Professional, enterprise-ready, calm tone."
        extra_tokens.value = "clean layout, soft studio lighting"
        extra_negative.value = "noisy, cluttered, distorted"
        override_style_tokens.value = "minimalism, modern corporate"

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
        if core_subject.value.strip():
            constraints.setdefault("locked_fields", {})
            constraints["locked_fields"]["core_subject"] = core_subject.value.strip()
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
        header = widgets.HBox([use_constraints, load_demo])
        section_constraints = widgets.VBox(
            [core_subject, must_include, avoid, style_bias, palette_bias, locked_fields, extra_instructions]
        )
        section_scene = widgets.VBox([extra_tokens, extra_negative, override_style_tokens])
        ui = widgets.VBox([header, section_constraints, section_scene])
        display(ui)
        return ui
