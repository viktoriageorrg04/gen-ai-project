import re
from typing import Dict, List, Optional


def _normalize_tokens(items: List[str]) -> List[str]:
    return [re.sub(r"\s+", " ", t.strip().lower()) for t in items if t and t.strip()]


def _extract_hexes(items: List[str]) -> List[str]:
    hexes = []
    for item in items:
        if not item:
            continue
        hexes.extend(re.findall(r"#[0-9a-fA-F]{6}", item))
    return [h.lower() for h in hexes]


def _tokenize_manifest(manifest: Dict) -> List[str]:
    fields = []
    fields.extend(manifest.get("key_features", []))
    fields.extend(manifest.get("color_palette", []))
    fields.extend(manifest.get("art_style_tokens", []))
    fields.append(manifest.get("core_subject", ""))
    fields.append(manifest.get("brand_vibe", ""))
    return _normalize_tokens(fields)


def _coverage_score(required: List[str], corpus: List[str]) -> Optional[Dict]:
    required = _normalize_tokens(required)
    if not required:
        return None
    corpus_text = " ".join(corpus)
    hits = [token for token in required if token in corpus_text]
    return {"required": required, "hits": hits, "ratio": len(hits) / len(required)}


def _jaccard(a: List[str], b: List[str]) -> Optional[float]:
    a_set, b_set = set(_normalize_tokens(a)), set(_normalize_tokens(b))
    if not a_set and not b_set:
        return None
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def compute_constraint_scores(manifest: Dict, constraints: Dict) -> Dict:
    manifest_tokens = _tokenize_manifest(manifest)
    must_include = constraints.get("must_include", [])
    avoid = constraints.get("avoid", [])
    style_bias = constraints.get("style_bias", [])
    palette_bias = constraints.get("palette_bias", [])

    coverage = _coverage_score(must_include, manifest_tokens)
    violations = _coverage_score(avoid, manifest_tokens)
    avoid_compliance = None
    if violations is not None:
        avoid_compliance = 1 - violations["ratio"]

    style_overlap = _jaccard(style_bias, manifest.get("art_style_tokens", []))

    manifest_palette = manifest.get("color_palette", [])
    bias_hex = _extract_hexes(palette_bias)
    manifest_hex = _extract_hexes(manifest_palette)
    if bias_hex or manifest_hex:
        palette_overlap = _jaccard(bias_hex, manifest_hex)
        palette_label = "Palette match (hex)"
    else:
        palette_overlap = _jaccard(palette_bias, manifest_palette)
        palette_label = "Palette Jaccard"

    return {
        "Must include coverage": coverage["ratio"] if coverage else None,
        "Avoid compliance": avoid_compliance,
        "Style Jaccard": style_overlap,
        palette_label: palette_overlap,
    }


def plot_constraint_scores(scores: Dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return

    labels = list(scores.keys())
    values = [scores[k] for k in labels]
    colors = ["#3B82F6" if v is not None else "#D1D5DB" for v in values]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bars = ax.bar(labels, [v or 0 for v in values], color=colors)
    ax.set_ylim(0, 1.15)
    ax.set_title("Constraint Alignment (Manifest vs Requirements)", pad=8)
    ax.set_ylabel("Score (0 to 1)")
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="x", labelrotation=18)

    for bar, value in zip(bars, values):
        if value is None:
            label = "N/A"
            y = 0.02
        else:
            label = f"{value:.2f}"
            y = min(value + 0.035, 1.02)
        ax.text(bar.get_x() + bar.get_width() / 2, y, label, ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    plt.show()
