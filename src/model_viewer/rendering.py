from __future__ import annotations

import html
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .diff import MappingRow, ModelDiff
from .formatting import format_bytes, format_params, json_dumps, markdown_table, pct_delta, shape_text
from .key_patterns import fold_key_patterns
from .schema import ModelSnapshot, TensorInfo, dtype_nbytes, normalize_dtype


VIEW_ORDER = ("overview", "heatmap", "detail", "mapping", "memory", "tree", "patterns", "blocks")
MODULES = (
    "embed",
    "ln1",
    "linear_attn",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "ln2",
    "gate",
    "up",
    "down",
    "vision",
    "mtp",
)
STATUS_GLYPHS = {
    "exact": "░",
    "equivalent": "▓",
    "different": "█",
    "left_only": "!",
    "right_only": "+",
    "auxiliary": "·",
    "none": " ",
}


def parse_views(raw: str) -> List[str]:
    if raw == "all":
        return list(VIEW_ORDER)
    views = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [view for view in views if view not in VIEW_ORDER and view != "json"]
    if invalid:
        raise ValueError(f"Unsupported view(s): {', '.join(invalid)}")
    return views


def render_show(snapshot: ModelSnapshot, views: Sequence[str], output_format: str, layer: Optional[int] = None) -> str:
    if output_format == "json" or "json" in views:
        return json_dumps(snapshot.to_dict())
    if output_format == "mermaid":
        return render_overview_mermaid(snapshot)
    if output_format == "drawio":
        if "patterns" in views:
            return render_key_patterns_drawio(snapshot)
        return render_overview_drawio(snapshot)

    sections: List[Tuple[str, str]] = []
    for view in views:
        if view == "overview":
            sections.append(("Overview", render_overview_mermaid(snapshot)))
        elif view == "detail":
            sections.append(("Layer Detail", render_layer_detail(snapshot, layer=layer or 0)))
        elif view == "memory":
            sections.append(("Memory Footprint", render_memory(snapshot)))
        elif view == "tree":
            sections.append(("Raw Tree", render_tree(snapshot)))
        elif view == "patterns":
            sections.append(("Safetensor Key Patterns", render_key_patterns(snapshot)))
        elif view == "blocks":
            sections.append(("Character Block Diagram", render_block_diagram(snapshot)))
        elif view in {"heatmap", "mapping"}:
            sections.append((view.title(), f"{view} is available for diff output."))
    return _join_sections(sections, output_format)


def render_diff(diff: ModelDiff, views: Sequence[str], output_format: str, layer: Optional[int] = None) -> str:
    if output_format == "json" or "json" in views:
        return json_dumps(
            {
                "left": diff.left.to_dict(),
                "right": diff.right.to_dict(),
                "summary": diff.summary(),
                "rows": [
                    {
                        "left": row.left_names,
                        "right": row.right_names,
                        "status": row.status,
                        "reason": row.reason,
                        "layer": row.layer,
                        "modules": row.modules,
                    }
                    for row in diff.rows
                ],
            }
        )
    if output_format == "mermaid":
        return render_diff_overview_mermaid(diff)
    if output_format == "drawio":
        if "patterns" in views:
            return render_diff_key_patterns_drawio(diff)
        return render_diff_drawio(diff)

    sections: List[Tuple[str, str]] = []
    for view in views:
        if view == "overview":
            sections.append(("Overview", render_diff_overview_mermaid(diff)))
        elif view == "heatmap":
            sections.append(("Heatmap", render_heatmap(diff)))
        elif view == "detail":
            sections.append(("Layer Detail", render_layer_detail(diff.left, diff.right, layer=layer or 0)))
        elif view == "mapping":
            sections.append(("Key Mapping", render_mapping(diff)))
        elif view == "memory":
            sections.append(("Memory Footprint", render_memory_diff(diff)))
        elif view == "tree":
            sections.append((f"Raw Tree: {diff.left.name}", render_tree(diff.left)))
            sections.append((f"Raw Tree: {diff.right.name}", render_tree(diff.right)))
        elif view == "patterns":
            sections.append((f"Safetensor Key Patterns: {diff.left.name}", render_key_patterns(diff.left)))
            sections.append((f"Safetensor Key Patterns: {diff.right.name}", render_key_patterns(diff.right)))
        elif view == "blocks":
            sections.append((f"Character Block Diagram: {diff.left.name}", render_block_diagram(diff.left)))
            sections.append((f"Character Block Diagram: {diff.right.name}", render_block_diagram(diff.right)))
    return _join_sections(sections, output_format)


def render_overview_mermaid(snapshot: ModelSnapshot) -> str:
    profile = snapshot.profile
    title = _model_title(snapshot)
    layers = int(profile.get("num_hidden_layers") or _infer_layer_count(snapshot))
    hidden = profile.get("hidden_size") or "?"
    dtype = _dominant_dtype(snapshot)
    vocab = profile.get("vocab_size") or "?"
    lm_head = "LM Head (tied)" if profile.get("tie_word_embeddings") else "LM Head"
    layer_kinds = _profile_layer_kinds(profile, layers)
    linear_layers = [idx for idx, kind in enumerate(layer_kinds) if _is_linear_attention(kind)]
    full_layers = [idx for idx, kind in enumerate(layer_kinds) if not _is_linear_attention(kind)]
    hybrid = bool(linear_layers and full_layers)
    lines = [
        "flowchart TB",
        f"    subgraph M[\"{_esc_mermaid(title)}\"]",
    ]
    if _multimodal_input_lines(snapshot, int(profile.get("hidden_size") or 0)):
        lines.append("        IN[\"Multimodal Router<br/>text + vision tokens\"]")
        lines.append(f"        VIT[\"Vision Encoder<br/>{_esc_mermaid(_vision_label(snapshot))}\"]")
        lines.append("        IN --> VIT")
        lines.append("        IN --> EMB")
        lines.append("        VIT --> EMB")
    lines.append(f"        EMB[\"Embedding<br/>[{vocab},{hidden}] {dtype}\"]")
    if hybrid:
        lines.extend([
            f"        SCHED[\"Hybrid Schedule<br/>{_esc_mermaid(_schedule_label(profile, layer_kinds))}\"]",
            f"        DELTA[\"DeltaNet / linear_attn x {len(linear_layers)}<br/>O(T), State Cache\"]",
            f"        GQA[\"GQA / self_attn x {len(full_layers)}<br/>{_esc_mermaid(_gqa_label(profile))}<br/>KV Cache + O(T^2)\"]",
            f"        FFN[\"{_esc_mermaid(_ffn_label(profile))}\"]",
            "        EMB --> SCHED",
            "        SCHED --> DELTA --> FFN",
            "        SCHED --> GQA --> FFN",
        ])
        tail = "FFN"
    else:
        lines.extend([
            f"        DEC[\"Decoder x {layers}<br/>{_esc_mermaid(_gqa_label(profile))}<br/>{_esc_mermaid(_ffn_label(profile))}\"]",
            "        EMB --> DEC",
        ])
        tail = "DEC"
    if _mtp_lines(snapshot):
        lines.append("        MTP[\"MTP Auxiliary Head<br/>side prediction\"]")
        lines.append(f"        {tail} --> MTP")
        tail = "MTP"
    lines.extend([
        "        NORM[\"Final Norm\"]",
        f"        LMH[\"{lm_head}\"]",
        f"        {tail} --> NORM --> LMH",
        "    end",
    ])
    return "\n".join(lines)


def render_diff_overview_mermaid(diff: ModelDiff) -> str:
    left = diff.left
    right = diff.right
    summary = diff.summary()
    left_profile = left.profile
    right_profile = right.profile
    changed = summary.get("different", 0) + summary.get("left_only", 0) + summary.get("right_only", 0)
    return f"""flowchart TB
    subgraph LEFT["{_esc_mermaid(_model_title(left))}"]
        L_CARD["{_esc_mermaid(_overview_card(left))}"]
        L_ARCH["{_esc_mermaid(_arch_label(left_profile))}"]
        L_CARD --> L_ARCH
    end
    subgraph RIGHT["{_esc_mermaid(_model_title(right))}"]
        R_CARD["{_esc_mermaid(_overview_card(right))}"]
        R_ARCH["{_esc_mermaid(_arch_label(right_profile))}"]
        R_CARD --> R_ARCH
    end
    L_CARD -.->|"params {pct_delta(left_profile.get('total_params') or left.total_params, right_profile.get('total_params') or right.total_params)}"| R_CARD
    L_ARCH -.->|"equiv {summary.get('equivalent', 0)} / changed {changed}"| R_ARCH"""


def render_overview_drawio(snapshot: ModelSnapshot) -> str:
    return _drawio_xml(_model_title(snapshot), _overview_nodes(snapshot))


def render_diff_drawio(diff: ModelDiff) -> str:
    summary = diff.summary()
    return _drawio_xml(
        f"{diff.left.name} vs {diff.right.name}",
        [
            (diff.left.name, _overview_card(diff.left)),
            ("Diff", f"exact={summary.get('exact', 0)}, equiv={summary.get('equivalent', 0)}, changed={summary.get('different', 0) + summary.get('left_only', 0) + summary.get('right_only', 0)}"),
            (diff.right.name, _overview_card(diff.right)),
        ],
    )


def _overview_nodes(snapshot: ModelSnapshot) -> List[Tuple[str, str]]:
    p = snapshot.profile
    layers = int(p.get("num_hidden_layers") or _infer_layer_count(snapshot))
    layer_kinds = _profile_layer_kinds(p, layers)
    linear_layers = [idx for idx, kind in enumerate(layer_kinds) if _is_linear_attention(kind)]
    full_layers = [idx for idx, kind in enumerate(layer_kinds) if not _is_linear_attention(kind)]
    nodes: List[Tuple[str, str]] = []
    if _multimodal_input_lines(snapshot, int(p.get("hidden_size") or 0)):
        nodes.append(("Multimodal Router", _vision_label(snapshot)))
    nodes.append(("Embedding", _shape_for_module(snapshot, "embed")))
    if linear_layers and full_layers:
        nodes.extend([
            ("Hybrid Schedule", _schedule_label(p, layer_kinds).replace("<br/>", "\n")),
            ("DeltaNet", f"x{len(linear_layers)}\nState Cache"),
            ("GQA", f"x{len(full_layers)}\n{_gqa_label(p).replace('<br/>', ' ')}"),
        ])
    else:
        nodes.append(("Decoder", f"x{layers}\n{_gqa_label(p).replace('<br/>', ' ')}"))
    nodes.append(("FFN", _ffn_label(p).replace("<br/>", "\n")))
    if _mtp_lines(snapshot):
        nodes.append(("MTP", "auxiliary side head"))
    nodes.append(("Final Norm", f"hidden={p.get('hidden_size') or '?'}"))
    nodes.append(("LM Head", "tied" if p.get("tie_word_embeddings") else _shape_for_module(snapshot, "lm_head")))
    return nodes


def _overview_card(snapshot: ModelSnapshot) -> str:
    p = snapshot.profile
    layers = int(p.get("num_hidden_layers") or _infer_layer_count(snapshot))
    return "<br/>".join([
        f"layers={layers} hidden={p.get('hidden_size') or '?'}",
        f"vocab={p.get('vocab_size') or '?'} dtype={_dominant_dtype(snapshot)}",
        f"params={format_params(p.get('total_params') or snapshot.total_params)}",
    ])


def _structural_compare_rows(left: ModelSnapshot, right: ModelSnapshot) -> List[List[str]]:
    rows = []
    for metric, getter in (
        ("model_type", lambda s: s.profile.get("model_type") or "?"),
        ("params", lambda s: format_params(s.profile.get("total_params") or s.total_params)),
        ("layers", lambda s: str(s.profile.get("num_hidden_layers") or _infer_layer_count(s))),
        ("hidden", lambda s: str(s.profile.get("hidden_size") or "?")),
        ("vocab", lambda s: str(s.profile.get("vocab_size") or "?")),
        ("attention", lambda s: _attention_summary(s)),
        ("ffn", lambda s: _ffn_label(s.profile).replace("<br/>", "; ")),
        ("cache", lambda s: _cache_summary(s)),
    ):
        rows.append([metric, getter(left), getter(right)])
    return rows


def _attention_summary(snapshot: ModelSnapshot) -> str:
    p = snapshot.profile
    layers = int(p.get("num_hidden_layers") or _infer_layer_count(snapshot))
    kinds = _profile_layer_kinds(p, layers)
    linear = [idx for idx, kind in enumerate(kinds) if _is_linear_attention(kind)]
    full = [idx for idx, kind in enumerate(kinds) if not _is_linear_attention(kind)]
    if linear and full:
        return f"DeltaNet={len(linear)} GQA={len(full)} macro={_macro_summary(kinds)}"
    return _gqa_label(p).replace("<br/>", "; ")


def _cache_summary(snapshot: ModelSnapshot) -> str:
    p = snapshot.profile
    layers = int(p.get("num_hidden_layers") or _infer_layer_count(snapshot))
    kinds = _profile_layer_kinds(p, layers)
    linear = sum(1 for kind in kinds if _is_linear_attention(kind))
    kv = int(p.get("num_kv_cache_layers") or layers - linear or layers)
    if linear:
        return f"KV={kv}; State={linear}"
    return f"KV={kv}"


def _schedule_label(profile: Dict[str, object], layer_kinds: Sequence[str]) -> str:
    linear = [idx for idx, kind in enumerate(layer_kinds) if _is_linear_attention(kind)]
    full = [idx for idx, kind in enumerate(layer_kinds) if not _is_linear_attention(kind)]
    labels = ["DeltaNet" if _is_linear_attention(kind) else "GQA" for kind in layer_kinds]
    period = _repeating_period(labels, max_period=16)
    pieces = [f"DeltaNet={len(linear)} GQA={len(full)}"]
    if period:
        macro = " + ".join(period)
        repeats = len(layer_kinds) // len(period)
        pieces.append(f"macro x{repeats}: {macro}")
    kv_layers = int(profile.get("num_kv_cache_layers") or len(full))
    pieces.append(f"KV layers={kv_layers} State layers={len(linear)}")
    return "<br/>".join(pieces)


def _macro_summary(layer_kinds: Sequence[str]) -> str:
    labels = ["DeltaNet" if _is_linear_attention(kind) else "GQA" for kind in layer_kinds]
    period = _repeating_period(labels, max_period=16)
    if not period:
        return "custom"
    return " + ".join(period)


def _gqa_label(profile: Dict[str, object]) -> str:
    heads = int(profile.get("num_attention_heads") or 0)
    kv_heads = int(profile.get("num_key_value_heads") or heads or 0)
    head_dim = int(profile.get("head_dim") or 0)
    ratio = (heads / kv_heads) if heads and kv_heads else 0
    label = f"GQA heads={heads or '?'} kv={kv_heads or '?'}"
    if ratio:
        label += f" group={ratio:g}"
    if head_dim:
        label += f" head_dim={head_dim}"
    kv_layers = profile.get("num_kv_cache_layers")
    if kv_layers:
        label += f"<br/>KV cache layers={kv_layers}"
    return label


def _group_ratio_text(profile: Dict[str, object]) -> str:
    heads = int(profile.get("num_attention_heads") or 0)
    kv_heads = int(profile.get("num_key_value_heads") or heads or 0)
    if not heads or not kv_heads:
        return "?"
    return f"{heads / kv_heads:g}"


def _ffn_label(profile: Dict[str, object]) -> str:
    experts = int(profile.get("num_experts") or 0)
    hidden = profile.get("hidden_size") or "?"
    if experts:
        active = profile.get("num_experts_per_tok") or "?"
        moe_i = profile.get("moe_intermediate_size") or "?"
        shared = int(profile.get("shared_expert_intermediate_size") or 0)
        suffix = f"<br/>shared={shared}" if shared else ""
        return f"SwiGLU MoE<br/>experts={experts} active={active} i_e={moe_i}{suffix}"
    return f"Dense SwiGLU<br/>hidden={hidden} intermediate={profile.get('intermediate_size') or '?'}"


def _vision_label(snapshot: ModelSnapshot) -> str:
    p = snapshot.profile
    vit = p.get("vit") if isinstance(p.get("vit"), dict) else {}
    vision = _config_section(snapshot.config, "vision_config")
    depth = vit.get("depth") or vision.get("depth") or "?"
    hidden = vit.get("hidden_size") or vision.get("hidden_size") or "?"
    heads = vit.get("num_heads") or vision.get("num_heads") or "?"
    out_hidden = vit.get("out_hidden_size") or vision.get("out_hidden_size") or p.get("hidden_size") or "?"
    return f"ViT x{depth}, hidden={hidden}, heads={heads}, out={out_hidden}"


def render_key_patterns(snapshot: ModelSnapshot, limit: int = 240) -> str:
    patterns = fold_key_patterns(snapshot)
    lines = [
        f"Safetensor Key Folding [{len(snapshot.tensors)} keys -> {len(patterns)} patterns]",
    ]
    hints = _pattern_architecture_hints(snapshot)
    if hints:
        lines.extend(hints)
    shown = patterns[:limit]
    for idx, pattern in enumerate(shown):
        branch = "└──" if idx == len(shown) - 1 and len(patterns) <= limit else "├──"
        suffix = f"x{pattern.count}"
        if pattern.shape:
            suffix += f"  {shape_text(pattern.shape)}"
        if pattern.dtype and pattern.dtype != "unknown":
            suffix += f"  {pattern.dtype}"
        lines.append(f"{branch} {pattern.pattern}  {suffix}")
    if len(patterns) > limit:
        lines.append(f"└── ... {len(patterns) - limit} more patterns")
    return "\n".join(lines)


def _pattern_architecture_hints(snapshot: ModelSnapshot) -> List[str]:
    p = snapshot.profile
    layers = int(p.get("num_hidden_layers") or _infer_layer_count(snapshot))
    kinds = _profile_layer_kinds(p, layers)
    if not any(_is_linear_attention(kind) for kind in kinds):
        return []
    linear = [idx for idx, kind in enumerate(kinds) if _is_linear_attention(kind)]
    full = [idx for idx, kind in enumerate(kinds) if not _is_linear_attention(kind)]
    return [
        f"Architecture: DeltaNet layers {_format_index_ranges(linear)}; GQA layers {_format_index_ranges(full)}",
        f"Expected key split: linear_attn.* x{len(linear)}; self_attn.* x{len(full)}",
    ]


def render_block_diagram(snapshot: ModelSnapshot) -> str:
    p = snapshot.profile
    dtype = _dominant_dtype(snapshot)
    layers = int(p.get("num_hidden_layers") or _infer_layer_count(snapshot))
    hidden = int(p.get("hidden_size") or 0)
    intermediate = int(p.get("intermediate_size") or 0)
    heads = int(p.get("num_attention_heads") or 0)
    kv_heads = int(p.get("num_key_value_heads") or heads or 0)
    head_dim = int(p.get("head_dim") or (hidden // max(1, heads)) if hidden else 0)
    q_dim = heads * head_dim if heads and head_dim else hidden
    kv_dim = kv_heads * head_dim if kv_heads and head_dim else hidden
    vocab = int(p.get("vocab_size") or 0)
    is_tied = bool(p.get("tie_word_embeddings"))
    num_experts = int(p.get("num_experts") or 0)
    experts_per_tok = int(p.get("num_experts_per_tok") or 0)
    moe_intermediate = int(p.get("moe_intermediate_size") or 0)
    shared_intermediate = int(p.get("shared_expert_intermediate_size") or 0)
    layer_kinds = _profile_layer_kinds(p, layers)
    linear_layers = [idx for idx, kind in enumerate(layer_kinds) if _is_linear_attention(kind)]
    full_layers = [idx for idx, kind in enumerate(layer_kinds) if not _is_linear_attention(kind)]
    hybrid = bool(linear_layers and full_layers)
    linear_key_heads = int(p.get("linear_num_key_heads") or 0)
    linear_value_heads = int(p.get("linear_num_value_heads") or 0)
    linear_key_dim = int(p.get("linear_key_head_dim") or 0)
    linear_value_dim = int(p.get("linear_value_head_dim") or 0)
    linear_qkv_shape = _first_shape_by_name(snapshot, ".linear_attn.in_proj_qkv.") or _shape_tuple(
        (linear_key_heads * linear_key_dim) + (2 * linear_value_heads * linear_value_dim),
        hidden,
    )
    linear_out_shape = _first_shape_by_name(snapshot, ".linear_attn.out_proj.") or _shape_tuple(
        hidden,
        linear_value_heads * linear_value_dim,
    )

    embed_shape = _first_shape(snapshot, "embed") or _shape_tuple(vocab, hidden)
    norm_shape = _first_shape(snapshot, "final_norm") or _shape_tuple(hidden)
    embed_name = _first_tensor_name(snapshot, "embed", "model.embed_tokens.weight")
    norm_name = _first_tensor_name(snapshot, "final_norm", "model.norm.weight")
    lm_head_name = _first_tensor_name(snapshot, "lm_head", "lm_head.weight")
    q_shape = _first_shape(snapshot, "q_proj") or _shape_tuple(q_dim, hidden)
    k_shape = _first_shape(snapshot, "k_proj") or _shape_tuple(kv_dim, hidden)
    v_shape = _first_shape(snapshot, "v_proj") or _shape_tuple(kv_dim, hidden)
    o_shape = _first_shape(snapshot, "o_proj") or _shape_tuple(hidden, q_dim)
    gate_shape = _first_shape(snapshot, "gate") or _shape_tuple(intermediate, hidden)
    up_shape = _first_shape(snapshot, "up") or _shape_tuple(intermediate, hidden)
    down_shape = _first_shape(snapshot, "down") or _shape_tuple(hidden, intermediate)
    lm_head_shape = _first_shape(snapshot, "lm_head") or _shape_tuple(vocab, hidden)

    title = f"{snapshot.name} [{p.get('model_type') or 'model'} | {format_params(p.get('total_params') or snapshot.total_params)} | {dtype}]"
    width = 88
    lines = [title, ""]
    multimodal = _multimodal_input_lines(snapshot, hidden)
    if multimodal:
        lines.extend([
            _box("MULTIMODAL INPUT ROUTER", multimodal, width=width),
            "        │",
            "        ▼",
        ])
    lines.extend([
        _box("TOKEN EMBEDDING", [
            f"{embed_name}  {shape_text(embed_shape)}  {dtype}",
            f"vocab={vocab or '?'}  hidden={hidden or '?'}",
        ], width=width),
        "        │",
        "        ▼",
    ])
    if hybrid:
        lines.extend([
            _box("HYBRID LAYER SCHEDULE", _layer_schedule_lines(
                layer_kinds=layer_kinds,
                linear_layers=linear_layers,
                full_layers=full_layers,
                kv_cache_layers=int(p.get("num_kv_cache_layers") or len(full_layers)),
                full_attention_interval=int(p.get("full_attention_interval") or 0),
            ), width=width),
            "        │",
            "        ▼",
        ])
    lines.extend([
        _box(f"LANGUAGE DECODER STACK x {layers}", _decoder_block_lines(
            hidden=hidden,
            heads=heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            linear_layers=linear_layers,
            full_layers=full_layers,
            q_shape=q_shape,
            k_shape=k_shape,
            v_shape=v_shape,
            o_shape=o_shape,
            linear_qkv_shape=linear_qkv_shape,
            linear_out_shape=linear_out_shape,
            linear_key_heads=linear_key_heads,
            linear_value_heads=linear_value_heads,
            linear_key_dim=linear_key_dim,
            linear_value_dim=linear_value_dim,
            attn_output_gate=bool(p.get("attn_output_gate")),
            gate_shape=gate_shape,
            up_shape=up_shape,
            down_shape=down_shape,
            num_experts=num_experts,
            experts_per_tok=experts_per_tok,
            moe_intermediate=moe_intermediate,
            shared_intermediate=shared_intermediate,
            dtype=dtype,
        ), width=width),
        "        │",
        "        ▼",
    ])
    mtp = _mtp_lines(snapshot)
    if mtp:
        lines.extend([
            _box("MTP AUXILIARY HEAD", mtp, width=width),
            "        │",
            "        ▼",
        ])
    lines.extend([
        _box("FINAL NORM", [
            f"{norm_name}  {shape_text(norm_shape)}  {dtype}",
        ], width=width),
        "        │",
        "        ▼",
        _box("LM HEAD", [
            "(tied with embedding)" if is_tied else f"{lm_head_name}  {shape_text(lm_head_shape)}  {dtype}",
            "checkpoint may still store lm_head.weight separately" if is_tied and _first_shape(snapshot, "lm_head") else "",
        ], width=width),
    ])
    return "\n".join(line for line in lines if line != "")


def render_key_patterns_drawio(snapshot: ModelSnapshot) -> str:
    nodes = _pattern_nodes(snapshot, limit=12)
    return _drawio_xml(f"Safetensor key folding: {snapshot.name}", nodes)


def render_diff_key_patterns_drawio(diff: ModelDiff) -> str:
    nodes = [
        (diff.left.name, f"{len(diff.left.tensors)} keys, {len(fold_key_patterns(diff.left))} patterns"),
        (diff.right.name, f"{len(diff.right.tensors)} keys, {len(fold_key_patterns(diff.right))} patterns"),
    ]
    return _drawio_xml(f"Safetensor key folding: {diff.left.name} vs {diff.right.name}", nodes)


def render_heatmap(diff: ModelDiff) -> str:
    matrix = _heatmap_matrix(diff)
    layers = sorted(layer for layer in matrix if layer is not None)
    lines = []
    lines.extend(_diff_schedule_lines(diff))
    header = "layer".ljust(12) + " ".join(module.rjust(8) for module in MODULES) + "   sum"
    lines.append(header)
    if None in matrix:
        lines.append(_heatmap_line("global", matrix[None]))
    for item in _fold_layers(layers, matrix):
        label, row = item
        lines.append(_heatmap_line(label, row))
    lines.append("")
    lines.append("Legend: ░ exact  ▓ equivalent(dtype/fused/tied)  █ real diff  + right-only  ! left-only")
    return "\n".join(lines)


def render_mapping(diff: ModelDiff, limit: int = 240) -> str:
    rows = []
    for row in diff.rows[:limit]:
        rows.append(
            [
                "\n".join(row.left_names) if row.left_names else "",
                "\n".join(row.right_names) if row.right_names else "",
                row.status,
                row.reason,
            ]
        )
    if len(diff.rows) > limit:
        rows.append(["...", "...", "truncated", f"{len(diff.rows) - limit} rows hidden"])
    summary = markdown_table(["Metric", diff.left.name, diff.right.name], _structural_compare_rows(diff.left, diff.right))
    mapping = markdown_table(["LEFT key", "RIGHT key", "MATCH", "DETAIL"], rows)
    return f"{summary}\n\n{mapping}" if summary else mapping


def render_layer_detail(left: ModelSnapshot, right: Optional[ModelSnapshot] = None, layer: int = 0) -> str:
    if right is None:
        return _single_layer_detail(left, layer)
    return _diff_layer_detail(left, right, layer)


def memory_summary(
    snapshot: ModelSnapshot,
    seq_len: Optional[int] = None,
    batch_size: int = 1,
    include_kv: bool = True,
) -> List[Dict[str, object]]:
    buckets = _memory_buckets(snapshot, seq_len=seq_len, batch_size=batch_size, include_kv=include_kv)
    return [{"bucket": name, "bytes": value} for name, value in buckets]


def render_memory(
    snapshot: ModelSnapshot,
    seq_len: Optional[int] = None,
    batch_size: int = 1,
    include_kv: bool = True,
) -> str:
    buckets = _memory_buckets(snapshot, seq_len=seq_len, batch_size=batch_size, include_kv=include_kv)
    rows = [[name, format_bytes(value)] for name, value in buckets]
    rows.append(["TOTAL", format_bytes(sum(value for _, value in buckets))])
    return markdown_table(["Bucket", "Memory"], rows)


def render_memory_diff(diff: ModelDiff, seq_len: Optional[int] = None, batch_size: int = 1) -> str:
    left = dict(_memory_buckets(diff.left, seq_len=seq_len, batch_size=batch_size, include_kv=True))
    right = dict(_memory_buckets(diff.right, seq_len=seq_len, batch_size=batch_size, include_kv=True))
    names = sorted(set(left) | set(right), key=_bucket_order)
    rows = []
    for name in names:
        l_value = left.get(name, 0.0)
        r_value = right.get(name, 0.0)
        rows.append([name, format_bytes(l_value), format_bytes(r_value), pct_delta(l_value, r_value)])
    rows.append([
        "TOTAL",
        format_bytes(sum(left.values())),
        format_bytes(sum(right.values())),
        pct_delta(sum(left.values()), sum(right.values())),
    ])
    return markdown_table(["Bucket", diff.left.name, diff.right.name, "Delta"], rows)


def render_tree(snapshot: ModelSnapshot) -> str:
    profile = snapshot.profile
    if profile:
        return _profile_tree(snapshot)
    return _tensor_tree(snapshot)


def _profile_tree(snapshot: ModelSnapshot) -> str:
    p = snapshot.profile
    name = p.get("model_type") or snapshot.name
    total = p.get("total_params") or snapshot.total_params
    hidden = p.get("hidden_size") or "?"
    layers = int(p.get("num_hidden_layers") or _infer_layer_count(snapshot))
    heads = p.get("num_attention_heads") or "?"
    kv_heads = p.get("num_key_value_heads") or heads
    intermediate = p.get("intermediate_size") or "?"
    vocab = p.get("vocab_size") or "?"
    dtype = _dominant_dtype(snapshot)
    layer_kinds = _profile_layer_kinds(p, layers)
    linear_layers = [idx for idx, kind in enumerate(layer_kinds) if _is_linear_attention(kind)]
    full_layers = [idx for idx, kind in enumerate(layer_kinds) if not _is_linear_attention(kind)]
    embed_name = _first_tensor_name(snapshot, "embed", "model.embed_tokens.weight").replace(".weight", "")
    norm_name = _first_tensor_name(snapshot, "final_norm", "model.norm.weight").replace(".weight", "")
    layer_prefix = _layer_prefix(snapshot)
    lines = [
        f"{name} [{format_params(total)} params, {dtype}]",
        f"├── {embed_name} [{vocab}, {hidden}]",
    ]
    multimodal = _multimodal_input_lines(snapshot, int(p.get("hidden_size") or 0))
    if multimodal:
        lines.extend([
            "├── multimodal",
            f"│   ├── vision_encoder {_vision_label(snapshot)}",
            "│   └── visual_merger -> language hidden",
        ])
    lines.append(f"├── {layer_prefix}.[0..{max(0, layers - 1)}] x {layers}")
    if linear_layers and full_layers:
        lines.extend([
            f"│   ├── layer_types {_schedule_label(p, layer_kinds).replace('<br/>', '; ')}",
            f"│   ├── DeltaNet layers {_format_index_ranges(linear_layers)}",
            "│   │   ├── linear_attn in_proj_qkv / in_proj_a / in_proj_b / in_proj_z",
            "│   │   ├── linear_attn conv1d / A_log / dt_bias / norm / out_proj",
            "│   │   └── State Cache (no [B,H,T,T] attention matrix)",
            f"│   ├── GQA layers {_format_index_ranges(full_layers)}",
            f"│   │   ├── self_attn q/k/v/o [heads={heads}, kv_heads={kv_heads}, group={_group_ratio_text(p)}]",
            f"│   │   └── KV Cache layers={p.get('num_kv_cache_layers') or len(full_layers)}",
        ])
    else:
        lines.extend([
            f"│   ├── self_attn q/k/v/o [heads={heads}, kv_heads={kv_heads}, group={_group_ratio_text(p)}]",
            f"│   ├── KV Cache layers={p.get('num_kv_cache_layers') or layers}",
        ])
    if int(p.get("num_experts") or 0):
        lines.extend([
            f"│   ├── moe router experts={p.get('num_experts')} active={p.get('num_experts_per_tok')}",
            f"│   ├── experts.[0..{int(p.get('num_experts') or 1) - 1}] gate/up/down [intermediate={p.get('moe_intermediate_size') or '?'}]",
        ])
        if int(p.get("shared_expert_intermediate_size") or 0):
            lines.append(f"│   ├── shared_expert [intermediate={p.get('shared_expert_intermediate_size')}]")
    else:
        lines.append(f"│   ├── dense_swiglu_mlp gate/up/down [intermediate={intermediate}]")
    lines.extend([
        "│   ├── input_layernorm",
        "│   └── post_attention_layernorm",
    ])
    if _mtp_lines(snapshot):
        lines.extend([
            "├── mtp",
            "│   ├── mtp.layers.[0..0] self_attn + dense_swiglu_mlp",
            "│   └── mtp.fc / mtp.norm / pre_fc_norm_*",
        ])
    lines.append(f"├── {norm_name} [{hidden}]")
    if p.get("tie_word_embeddings"):
        lines.append("└── lm_head (tied with embedding)")
    else:
        lines.append(f"└── lm_head [{vocab}, {hidden}]")
    return "\n".join(lines)


def _tensor_tree(snapshot: ModelSnapshot, limit: int = 120) -> str:
    lines = [f"{snapshot.name} [{format_params(snapshot.total_params)} params]"]
    for tensor in snapshot.tensors[:limit]:
        lines.append(f"├── {tensor.name} {shape_text(tensor.shape)} {tensor.dtype} {format_params(tensor.numel)}")
    if len(snapshot.tensors) > limit:
        lines.append(f"└── ... {len(snapshot.tensors) - limit} more tensors")
    return "\n".join(lines)


def _single_layer_detail(snapshot: ModelSnapshot, layer: int) -> str:
    tensors = [tensor for tensor in snapshot.tensors if tensor.layer == layer]
    by_module = defaultdict(list)
    for tensor in tensors:
        by_module[tensor.module].append(tensor)
    p = snapshot.profile
    layers = int(p.get("num_hidden_layers") or _infer_layer_count(snapshot))
    layer_kinds = _profile_layer_kinds(p, layers)
    kind = layer_kinds[layer] if 0 <= layer < len(layer_kinds) else "full_attention"
    hidden = int(p.get("hidden_size") or 0)
    heads = int(p.get("num_attention_heads") or 0)
    kv_heads = int(p.get("num_key_value_heads") or heads or 0)
    head_dim = int(p.get("head_dim") or (hidden // max(1, heads)) if hidden else 0)
    q_dim = heads * head_dim if heads and head_dim else hidden
    kv_dim = kv_heads * head_dim if kv_heads and head_dim else hidden
    intermediate = int(p.get("intermediate_size") or 0)
    linear_key_heads = int(p.get("linear_num_key_heads") or 0)
    linear_value_heads = int(p.get("linear_num_value_heads") or 0)
    linear_key_dim = int(p.get("linear_key_head_dim") or 0)
    linear_value_dim = int(p.get("linear_value_head_dim") or 0)
    dtype = _dominant_dtype(snapshot)
    q_shape = _module_shapes_or(by_module, "q_proj", _shape_tuple(q_dim, hidden), dtype)
    k_shape = _module_shapes_or(by_module, "k_proj", _shape_tuple(kv_dim, hidden), dtype)
    v_shape = _module_shapes_or(by_module, "v_proj", _shape_tuple(kv_dim, hidden), dtype)
    o_shape = _module_shapes_or(by_module, "o_proj", _shape_tuple(hidden, q_dim), dtype)
    linear_qkv = _module_shapes_or(
        by_module,
        "qkv_proj",
        _shape_tuple((linear_key_heads * linear_key_dim) + (2 * linear_value_heads * linear_value_dim), hidden),
        dtype,
    )
    ln_shape = _module_shapes_or(by_module, "ln1", _shape_tuple(hidden), dtype)
    post_ln_shape = _module_shapes_or(by_module, "ln2", _shape_tuple(hidden), dtype)
    gate_shape = _module_shapes_or(by_module, "gate", _shape_tuple(intermediate, hidden), dtype)
    up_shape = _module_shapes_or(by_module, "up", _shape_tuple(intermediate, hidden), dtype)
    down_shape = _module_shapes_or(by_module, "down", _shape_tuple(hidden, intermediate), dtype)
    lines = [
        f"LANGUAGE LAYER {layer} [{_layer_kind_label(kind)}]",
        f"  hidden={p.get('hidden_size', '?')} heads={p.get('num_attention_heads', '?')} kv_heads={p.get('num_key_value_heads', '?')} group={_group_ratio_text(p)}",
        "  input",
        f"    -> RMSNorm {ln_shape}",
    ]
    if _is_linear_attention(kind):
        lines.extend([
            f"    -> DeltaNet linear_attn in_proj_qkv={linear_qkv}",
            "       in_proj_a / in_proj_b / in_proj_z + conv1d + A_log + dt_bias",
            f"       linear_heads k={linear_key_heads or '?'}x{linear_key_dim or '?'} v={linear_value_heads or '?'}x{linear_value_dim or '?'}",
            "       no [B,H,T,T] attention matrix; contributes State Cache",
        ])
    else:
        lines.extend([
            f"    -> GQA self_attn q={q_shape} k={k_shape} v={v_shape} o={o_shape}",
            "       q_norm / k_norm + RoPE; contributes KV Cache and O(T^2) workspace",
        ])
    lines.extend([
        "    -> Residual Add",
        f"    -> RMSNorm {post_ln_shape}",
    ])
    if int(p.get("num_experts") or 0):
        lines.extend([
            f"    -> SwiGLU MoE router experts={p.get('num_experts')} active={p.get('num_experts_per_tok')}",
            f"       experts.[0..{int(p.get('num_experts') or 1) - 1}] gate/up/down intermediate={p.get('moe_intermediate_size') or '?'}",
        ])
    else:
        lines.append(f"    -> Dense SwiGLU MLP gate={gate_shape} up={up_shape} down={down_shape}")
    lines.append("    -> residual output")
    return "\n".join(lines)


def _diff_layer_detail(left: ModelSnapshot, right: ModelSnapshot, layer: int) -> str:
    left_modules = _layer_module_shapes(left, layer)
    right_modules = _layer_module_shapes(right, layer)
    rows = [
        ["layer_kind", _layer_kind_at(left, layer), _layer_kind_at(right, layer)],
        ["architecture", _arch_label(left.profile), _arch_label(right.profile)],
    ]
    for module in MODULES:
        if module == "embed":
            continue
        rows.append([module, left_modules.get(module, "-"), right_modules.get(module, "-")])
    return markdown_table(["Module", left.name, right.name], rows)


def _memory_buckets(
    snapshot: ModelSnapshot,
    seq_len: Optional[int] = None,
    batch_size: int = 1,
    include_kv: bool = True,
) -> List[Tuple[str, float]]:
    profile_buckets = _profile_weight_buckets(snapshot)
    tensor_params = snapshot.total_params
    profile_params = float(snapshot.profile.get("total_params") or 0)
    use_profile = bool(profile_buckets and (tensor_params == 0 or tensor_params < profile_params * 0.5 or _is_hybrid_snapshot(snapshot)))
    if use_profile:
        buckets = defaultdict(float, profile_buckets)
        if include_kv:
            kv_cache = _estimate_kv_cache(snapshot, seq_len=seq_len, batch_size=batch_size)
            if kv_cache:
                buckets["KV Cache"] += kv_cache
            state_cache = _estimate_state_cache(snapshot, batch_size=batch_size)
            if state_cache:
                buckets["State Cache"] += state_cache
        return [(name, buckets[name]) for name in sorted(buckets, key=_bucket_order)]

    buckets: Dict[str, float] = defaultdict(float)
    for tensor in snapshot.tensors:
        module = tensor.module
        if module == "embed":
            bucket = "Embedding"
        elif module in {"q_proj", "k_proj", "v_proj", "qkv_proj", "o_proj", "linear_attn"} or ".linear_attn." in tensor.name:
            bucket = "Attention"
        elif module in {"gate", "up", "gate_up", "down"} or tensor.kind in {"expert", "router"}:
            bucket = "MLP / MoE"
        elif module == "lm_head":
            bucket = "LM Head"
        elif module == "vision" or ".visual." in tensor.name:
            bucket = "Vision Encoder"
        elif module == "mtp" or tensor.name.startswith("mtp."):
            bucket = "MTP"
        elif tensor.kind == "norm" or module in {"ln1", "ln2", "final_norm"}:
            bucket = "Norms"
        elif tensor.kind == "quant_aux":
            bucket = "Quant Aux"
        else:
            bucket = "Other"
        buckets[bucket] += tensor.logical_bytes
    kv_cache = _estimate_kv_cache(snapshot, seq_len=seq_len, batch_size=batch_size) if include_kv else 0.0
    if kv_cache:
        buckets["KV Cache"] += kv_cache
    state_cache = _estimate_state_cache(snapshot, batch_size=batch_size) if include_kv else 0.0
    if state_cache:
        buckets["State Cache"] += state_cache
    return [(name, buckets[name]) for name in sorted(buckets, key=_bucket_order)]


def _profile_weight_buckets(snapshot: ModelSnapshot) -> Dict[str, float]:
    p = snapshot.profile
    layers = int(p.get("num_hidden_layers") or _infer_layer_count(snapshot))
    hidden = int(p.get("hidden_size") or 0)
    vocab = int(p.get("vocab_size") or 0)
    heads = int(p.get("num_attention_heads") or 0)
    kv_heads = int(p.get("num_key_value_heads") or heads or 0)
    head_dim = int(p.get("head_dim") or (hidden // max(1, heads)) if hidden else 0)
    intermediate = int(p.get("intermediate_size") or 0)
    if not (layers and hidden):
        return {}

    dtype = _dominant_dtype(snapshot)
    nbytes = dtype_nbytes(dtype) or 2.0
    layer_kinds = _profile_layer_kinds(p, layers)
    linear_count = sum(1 for kind in layer_kinds if _is_linear_attention(kind))
    full_count = layers - linear_count
    q_dim = heads * head_dim if heads and head_dim else hidden
    kv_dim = kv_heads * head_dim if kv_heads and head_dim else hidden
    standard_attn = int(p.get("standard_attention_params_per_layer") or 0)
    if not standard_attn:
        standard_attn = hidden * (q_dim + 2 * kv_dim) + q_dim * hidden
    linear_attn = int(p.get("linear_attention_params_per_layer") or 0)
    if not linear_attn:
        linear_attn = standard_attn

    experts = int(p.get("num_experts") or 0)
    moe_i = int(p.get("moe_intermediate_size") or 0)
    shared = int(p.get("shared_expert_intermediate_size") or 0)
    if experts and moe_i:
        mlp_per_layer = experts * 3 * hidden * moe_i + hidden * experts + 3 * hidden * shared
    else:
        mlp_per_layer = 3 * hidden * intermediate

    buckets: Dict[str, float] = defaultdict(float)
    buckets["Embedding"] = vocab * hidden * nbytes
    buckets["Attention"] = (full_count * standard_attn + linear_count * linear_attn) * nbytes
    buckets["MLP / MoE"] = layers * mlp_per_layer * nbytes
    buckets["Norms"] = (2 * layers * hidden + hidden) * nbytes
    if not p.get("tie_word_embeddings"):
        buckets["LM Head"] = vocab * hidden * nbytes
    vit = p.get("vit") if isinstance(p.get("vit"), dict) else {}
    vit_params = int(vit.get("params") or 0)
    if vit_params:
        buckets["Vision Encoder"] = vit_params * nbytes
    return buckets


def _estimate_kv_cache(snapshot: ModelSnapshot, seq_len: Optional[int], batch_size: int) -> float:
    p = snapshot.profile
    layers = int(p.get("num_kv_cache_layers") or p.get("num_hidden_layers") or 0)
    kv_heads = int(p.get("num_key_value_heads") or p.get("num_attention_heads") or 0)
    head_dim = int(p.get("head_dim") or 0)
    if not layers or not kv_heads or not head_dim:
        return 0.0
    length = int(seq_len or p.get("default_max_model_len") or 4096)
    dtype = _dominant_dtype(snapshot)
    nbytes = dtype_nbytes(dtype) or 2.0
    return float(batch_size * length * layers * 2 * kv_heads * head_dim * nbytes)


def _estimate_state_cache(snapshot: ModelSnapshot, batch_size: int) -> float:
    p = snapshot.profile
    layers = int(p.get("num_linear_attn_layers") or 0)
    if not layers:
        return 0.0
    heads = int(p.get("linear_num_key_heads") or p.get("num_attention_heads") or 0)
    key_dim = int(p.get("linear_key_head_dim") or p.get("head_dim") or 0)
    value_dim = int(p.get("linear_value_head_dim") or key_dim or 0)
    state_dim = max(key_dim, value_dim)
    if not heads or not state_dim:
        return 0.0
    dtype = _dominant_dtype(snapshot)
    nbytes = dtype_nbytes(dtype) or 2.0
    return float(batch_size * layers * heads * state_dim * state_dim * nbytes * 2)


def _heatmap_matrix(diff: ModelDiff) -> Dict[Optional[int], Dict[str, str]]:
    matrix: Dict[Optional[int], Dict[str, str]] = defaultdict(lambda: {module: "none" for module in MODULES})
    for row in diff.rows:
        target_layer = row.layer
        modules = _expand_modules(row.modules)
        if not modules:
            continue
        for module in modules:
            if module not in MODULES:
                continue
            old = matrix[target_layer][module]
            matrix[target_layer][module] = _worse_status(old, row.status)
    return matrix


def _diff_schedule_lines(diff: ModelDiff) -> List[str]:
    lines = []
    for label, snapshot in (("left", diff.left), ("right", diff.right)):
        p = snapshot.profile
        layers = int(p.get("num_hidden_layers") or _infer_layer_count(snapshot))
        kinds = _profile_layer_kinds(p, layers)
        if any(_is_linear_attention(kind) for kind in kinds):
            linear = [idx for idx, kind in enumerate(kinds) if _is_linear_attention(kind)]
            full = [idx for idx, kind in enumerate(kinds) if not _is_linear_attention(kind)]
            lines.append(
                f"{label} layer_types: DeltaNet {_format_index_ranges(linear)}; GQA {_format_index_ranges(full)}"
            )
    if lines:
        lines.append("")
    return lines


def _expand_modules(modules: Iterable[str]) -> List[str]:
    result: List[str] = []
    for module in modules:
        if module == "qkv_proj":
            result.extend(["q_proj", "k_proj", "v_proj"])
        elif module == "gate_up":
            result.extend(["gate", "up"])
        elif module == "final_norm":
            result.append("ln2")
        else:
            result.append(module)
    return result


def _worse_status(old: str, new: str) -> str:
    rank = {"none": 0, "auxiliary": 1, "exact": 2, "equivalent": 3, "right_only": 4, "left_only": 4, "different": 5}
    return new if rank.get(new, 0) >= rank.get(old, 0) else old


def _fold_layers(layers: List[int], matrix: Dict[Optional[int], Dict[str, str]]) -> List[Tuple[str, Dict[str, str]]]:
    if not layers:
        return []
    result: List[Tuple[str, Dict[str, str]]] = []
    start = prev = layers[0]
    start_row = matrix[start]
    for layer in layers[1:]:
        if layer == prev + 1 and matrix[layer] == start_row:
            prev = layer
            continue
        result.append((_layer_label(start, prev), start_row))
        start = prev = layer
        start_row = matrix[layer]
    result.append((_layer_label(start, prev), start_row))
    return result


def _layer_label(start: int, end: int) -> str:
    return f"layer {start}" if start == end else f"layer {start}~{end}"


def _heatmap_line(label: str, row: Dict[str, str]) -> str:
    cells = [STATUS_GLYPHS.get(row.get(module, "none"), "?").center(8) for module in MODULES]
    statuses = [row.get(module, "none") for module in MODULES]
    summary = "OK" if all(status in {"exact", "none"} for status in statuses) else "WARN"
    if any(status in {"different", "left_only", "right_only"} for status in statuses):
        summary = "DIFF"
    return label.ljust(12) + " ".join(cells) + f"   {summary}"


def _module_shapes(by_module: Dict[str, List[TensorInfo]], module: str) -> str:
    tensors = by_module.get(module) or []
    if not tensors:
        return "-"
    return ", ".join(f"{shape_text(tensor.shape)} {tensor.dtype}" for tensor in tensors[:3])


def _module_shapes_or(
    by_module: Dict[str, List[TensorInfo]],
    module: str,
    fallback_shape: Tuple[int, ...],
    fallback_dtype: str,
) -> str:
    tensors = by_module.get(module) or []
    known = [tensor for tensor in tensors if tensor.shape]
    if known:
        return ", ".join(f"{shape_text(tensor.shape)} {tensor.dtype}" for tensor in known[:3])
    return f"{shape_text(fallback_shape)} {fallback_dtype}"


def _first_shape(snapshot: ModelSnapshot, module: str) -> Tuple[int, ...]:
    tensor = next((item for item in snapshot.tensors if item.module == module and item.shape), None)
    return tensor.shape if tensor is not None else ()


def _first_tensor_name(snapshot: ModelSnapshot, module: str, fallback: str) -> str:
    tensor = next((item for item in snapshot.tensors if item.module == module), None)
    return tensor.name if tensor is not None else fallback


def _layer_prefix(snapshot: ModelSnapshot) -> str:
    tensor = next((item for item in snapshot.tensors if item.layer is not None and ".layers." in item.name), None)
    if tensor is None:
        return "model.layers"
    return tensor.name.split(".layers.", 1)[0] + ".layers"


def _first_shape_by_name(snapshot: ModelSnapshot, marker: str) -> Tuple[int, ...]:
    tensor = next((item for item in snapshot.tensors if marker in item.name and item.shape), None)
    return tensor.shape if tensor is not None else ()


def _shape_tuple(*values: int) -> Tuple[int, ...]:
    if not values or any(not value for value in values):
        return ()
    return tuple(int(value) for value in values)


def _profile_layer_kinds(profile: Dict[str, object], layers: int) -> List[str]:
    raw = profile.get("layer_kinds") or profile.get("layer_types")
    if isinstance(raw, list) and raw:
        kinds = [
            "linear_attention" if _is_linear_attention(str(item)) else "full_attention"
            for item in raw
        ]
        if len(kinds) < layers:
            kinds.extend(["full_attention"] * (layers - len(kinds)))
        return kinds[:layers]
    linear_count = int(profile.get("num_linear_attn_layers") or 0)
    if linear_count and layers:
        full_interval = int(profile.get("full_attention_interval") or 0)
        if full_interval > 1:
            return [
                "full_attention" if (idx + 1) % full_interval == 0 else "linear_attention"
                for idx in range(layers)
            ]
    return ["full_attention"] * layers


def _is_linear_attention(kind: object) -> bool:
    value = str(kind).lower()
    return "linear" in value or "delta" in value


def _layer_kind_label(kind: object) -> str:
    return "DeltaNet / linear_attention" if _is_linear_attention(kind) else "GQA / full_attention"


def _layer_kind_at(snapshot: ModelSnapshot, layer: int) -> str:
    layers = int(snapshot.profile.get("num_hidden_layers") or _infer_layer_count(snapshot))
    kinds = _profile_layer_kinds(snapshot.profile, layers)
    if 0 <= layer < len(kinds):
        return _layer_kind_label(kinds[layer])
    return "?"


def _is_hybrid_snapshot(snapshot: ModelSnapshot) -> bool:
    p = snapshot.profile
    layers = int(p.get("num_hidden_layers") or _infer_layer_count(snapshot))
    kinds = _profile_layer_kinds(p, layers)
    return any(_is_linear_attention(kind) for kind in kinds) and any(not _is_linear_attention(kind) for kind in kinds)


def _layer_schedule_lines(
    layer_kinds: Sequence[str],
    linear_layers: Sequence[int],
    full_layers: Sequence[int],
    kv_cache_layers: int,
    full_attention_interval: int,
) -> List[str]:
    layers = len(layer_kinds)
    linear_count = len(linear_layers)
    full_count = len(full_layers)
    full_share = (full_count / layers * 100.0) if layers else 0.0
    labels = ["DeltaNet" if _is_linear_attention(kind) else "GQA" for kind in layer_kinds]
    period = _repeating_period(labels, max_period=16)
    lines = [
        f"layer_types: {layers} layers, 0-based index",
        f"DeltaNet/linear={linear_count}  GQA/full={full_count}  O(T^2) share={full_share:.1f}%",
    ]
    if period:
        repeats = layers // len(period)
        macro = " -> ".join(f"L{idx + 1}:{label}" for idx, label in enumerate(period))
        lines.append(f"macro-block x{repeats}: [{macro}]")
    if full_attention_interval:
        lines.append(f"full_attention_interval={full_attention_interval}")
    lines.extend([
        f"DeltaNet layers: {_format_index_ranges(linear_layers)}",
        f"GQA layers: {_format_index_ranges(full_layers)}",
        f"KV Cache layers={kv_cache_layers or full_count}; State Cache layers={linear_count}",
        "Only GQA layers materialize O(T^2) attention workspace; DeltaNet keeps O(T) state.",
    ])
    return lines


def _repeating_period(items: Sequence[str], max_period: int = 16) -> List[str]:
    if not items:
        return []
    limit = min(max_period, len(items))
    for size in range(1, limit + 1):
        if len(items) % size:
            continue
        period = list(items[:size])
        if all(item == period[idx % size] for idx, item in enumerate(items)):
            return period
    return []


def _format_index_ranges(indices: Sequence[int], max_len: int = 68) -> str:
    if not indices:
        return "-"
    ranges = []
    start = prev = int(indices[0])
    for raw in indices[1:]:
        value = int(raw)
        if value == prev + 1:
            prev = value
            continue
        ranges.append(_range_text(start, prev))
        start = prev = value
    ranges.append(_range_text(start, prev))
    text = "{" + ",".join(ranges) + "}"
    if len(text) <= max_len:
        return text
    visible: List[str] = []
    for item in ranges:
        candidate = "{" + ",".join(visible + [item]) + ",...}"
        if len(candidate) > max_len:
            break
        visible.append(item)
    return "{" + ",".join(visible) + ",...}"


def _range_text(start: int, end: int) -> str:
    return str(start) if start == end else f"{start}..{end}"


def _multimodal_input_lines(snapshot: ModelSnapshot, hidden: int) -> List[str]:
    p = snapshot.profile
    vit = p.get("vit") if isinstance(p.get("vit"), dict) else {}
    vision = _config_section(snapshot.config, "vision_config")
    has_visual_tensors = any(tensor.name.startswith("model.visual.") for tensor in snapshot.tensors)
    if not (p.get("is_multimodal") or vit or vision or has_visual_tensors):
        return []

    depth = vit.get("depth") or vision.get("depth") or "?"
    v_hidden = vit.get("hidden_size") or vision.get("hidden_size") or "?"
    v_heads = vit.get("num_heads") or vision.get("num_heads") or "?"
    v_intermediate = vit.get("intermediate_size") or vision.get("intermediate_size") or "?"
    out_hidden = vit.get("out_hidden_size") or vision.get("out_hidden_size") or hidden or "?"
    patch = vit.get("patch_size") or vision.get("patch_size") or "?"
    temporal = vit.get("temporal_patch_size") or vision.get("temporal_patch_size") or "?"
    merge = vit.get("spatial_merge_size") or vision.get("spatial_merge_size") or "?"
    lines = [
        "text path: input_ids -> language_model.embed_tokens",
        f"vision path: patch_embed(patch={patch}, temporal={temporal}, merge={merge})",
        f"ViT blocks x {depth}: hidden={v_hidden} heads={v_heads} intermediate={v_intermediate}",
        f"visual merger: vision hidden {v_hidden} -> language hidden {out_hidden}",
    ]
    image_id = snapshot.config.get("image_token_id")
    video_id = snapshot.config.get("video_token_id")
    if image_id is not None or video_id is not None:
        lines.append(f"special tokens: image={image_id or '?'} video={video_id or '?'}")
    return lines


def _mtp_lines(snapshot: ModelSnapshot) -> List[str]:
    p = snapshot.profile
    count = int(p.get("mtp_num_layers") or 0)
    if not count:
        text_config = _config_section(snapshot.config, "text_config")
        count = int(text_config.get("mtp_num_hidden_layers") or 0)
    has_mtp_tensors = any(tensor.name.startswith("mtp.") for tensor in snapshot.tensors)
    if not count and not has_mtp_tensors:
        return []
    return [
        f"mtp.layers x {count or '?'}: self_attn + SwiGLU MLP + RMSNorm",
        "mtp.fc / mtp.norm / pre_fc_norm_* form a side prediction head",
        "MTP is separate from the main language decoder depth above.",
    ]


def _config_section(config: Dict[str, object], key: str) -> Dict[str, object]:
    value = config.get(key)
    return value if isinstance(value, dict) else {}


def _decoder_block_lines(
    hidden: int,
    heads: int,
    kv_heads: int,
    head_dim: int,
    linear_layers: Sequence[int],
    full_layers: Sequence[int],
    q_shape: Tuple[int, ...],
    k_shape: Tuple[int, ...],
    v_shape: Tuple[int, ...],
    o_shape: Tuple[int, ...],
    linear_qkv_shape: Tuple[int, ...],
    linear_out_shape: Tuple[int, ...],
    linear_key_heads: int,
    linear_value_heads: int,
    linear_key_dim: int,
    linear_value_dim: int,
    attn_output_gate: bool,
    gate_shape: Tuple[int, ...],
    up_shape: Tuple[int, ...],
    down_shape: Tuple[int, ...],
    num_experts: int,
    experts_per_tok: int,
    moe_intermediate: int,
    shared_intermediate: int,
    dtype: str,
) -> List[str]:
    hybrid = bool(linear_layers and full_layers)
    gqa_ratio = (heads / kv_heads) if heads and kv_heads else 0
    lines = [
        f"input hidden state: B x T x {hidden or '?'}",
        "│",
        "├─ RMSNorm: input_layernorm.weight",
        "│",
    ]
    if hybrid:
        lines.extend([
            "├─ Attention dispatch by layer_types",
            f"│  ├─ DeltaNet / linear_attn x {len(linear_layers)}",
            f"│  │  ├─ in_proj_qkv  {shape_text(linear_qkv_shape)}",
            "│  │  ├─ in_proj_a / in_proj_b / in_proj_z + conv1d + A_log + dt_bias",
            f"│  │  ├─ linear heads: k={linear_key_heads or '?'}x{linear_key_dim or '?'}  v={linear_value_heads or '?'}x{linear_value_dim or '?'}",
            f"│  │  ├─ out_proj {shape_text(linear_out_shape)}  attn_output_gate={_yes_no(attn_output_gate)}",
            "│  │  └─ no [B,H,T,T]; O(T) state path, State Cache at inference",
            f"│  └─ GQA / self_attn x {len(full_layers)}",
            f"│     ├─ q_proj  {shape_text(q_shape)}  heads={heads or '?'}",
            f"│     ├─ k_proj  {shape_text(k_shape)}  kv_heads={kv_heads or '?'}",
            f"│     ├─ v_proj  {shape_text(v_shape)}  head_dim={head_dim or '?'}",
            f"│     ├─ q_norm / k_norm + RoPE  group_ratio={gqa_ratio:g}" if gqa_ratio else "│     ├─ q_norm / k_norm + RoPE",
            f"│     └─ o_proj  {shape_text(o_shape)}; only branch with KV Cache and O(T^2)",
        ])
    else:
        lines.extend([
            "├─ Attention",
            f"│  ├─ q_proj  {shape_text(q_shape)}  heads={heads or '?'}",
            f"│  ├─ k_proj  {shape_text(k_shape)}  kv_heads={kv_heads or '?'}",
            f"│  ├─ v_proj  {shape_text(v_shape)}  head_dim={head_dim or '?'}",
            f"│  ├─ q_norm / k_norm  {dtype}",
            f"│  └─ o_proj  {shape_text(o_shape)}",
        ])
    lines.extend([
        "│",
        "├─ Residual Add",
        "│",
        "├─ RMSNorm: post_attention_layernorm.weight",
        "│",
    ])
    if num_experts:
        lines.extend([
            "├─ SwiGLU MoE MLP (same FFN on DeltaNet and GQA layers)",
            f"│  ├─ router gate.weight  experts={num_experts}  logits=B x T x {num_experts}",
            f"│  ├─ Top-K dispatch: active={experts_per_tok or '?'} experts/token",
            f"│  ├─ experts.{{0..{max(0, num_experts - 1)}}}.gate/up  [{moe_intermediate or '?'},{hidden or '?'}]",
            f"│  ├─ experts.{{0..{max(0, num_experts - 1)}}}.down     [{hidden or '?'},{moe_intermediate or '?'}]",
            f"│  └─ shared_expert intermediate={shared_intermediate}" if shared_intermediate else "│  └─ no shared_expert configured",
        ])
    else:
        lines.extend([
            "├─ Dense SwiGLU MLP" if hybrid else "├─ MLP",
            f"│  ├─ gate_proj  {shape_text(gate_shape)}",
            f"│  ├─ up_proj    {shape_text(up_shape)}",
            f"│  └─ down_proj  {shape_text(down_shape)}",
        ])
    lines.extend([
        "│",
        "└─ Residual Add -> next layer",
    ])
    return lines


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _box(title: str, body: Sequence[str], width: int = 72) -> str:
    top = "┌" + "─" * (width - 2) + "┐"
    bottom = "└" + "─" * (width - 2) + "┘"
    title_line = _box_line(title, width)
    content = [top, title_line, "├" + "─" * (width - 2) + "┤"]
    for line in body:
        if line:
            content.append(_box_line(line, width))
    content.append(bottom)
    return "\n".join(content)


def _box_line(text: str, width: int) -> str:
    value = str(text)
    available = width - 4
    if len(value) > available:
        value = value[: max(0, available - 3)] + "..."
    return "│ " + value.ljust(available) + " │"


def _layer_module_shapes(snapshot: ModelSnapshot, layer: int) -> Dict[str, str]:
    by_module: Dict[str, List[TensorInfo]] = defaultdict(list)
    for tensor in snapshot.tensors:
        if tensor.layer == layer:
            by_module[tensor.module].append(tensor)
    return {module: _layer_module_shape(snapshot, by_module, module) for module in by_module}


def _layer_module_shape(snapshot: ModelSnapshot, by_module: Dict[str, List[TensorInfo]], module: str) -> str:
    p = snapshot.profile
    hidden = int(p.get("hidden_size") or 0)
    heads = int(p.get("num_attention_heads") or 0)
    kv_heads = int(p.get("num_key_value_heads") or heads or 0)
    head_dim = int(p.get("head_dim") or (hidden // max(1, heads)) if hidden else 0)
    intermediate = int(p.get("intermediate_size") or 0)
    q_dim = heads * head_dim if heads and head_dim else hidden
    kv_dim = kv_heads * head_dim if kv_heads and head_dim else hidden
    linear_key_heads = int(p.get("linear_num_key_heads") or 0)
    linear_value_heads = int(p.get("linear_num_value_heads") or 0)
    linear_key_dim = int(p.get("linear_key_head_dim") or 0)
    linear_value_dim = int(p.get("linear_value_head_dim") or 0)
    linear_qkv = (linear_key_heads * linear_key_dim) + (2 * linear_value_heads * linear_value_dim)
    fallback = {
        "ln1": _shape_tuple(hidden),
        "ln2": _shape_tuple(hidden),
        "q_proj": _shape_tuple(q_dim, hidden),
        "k_proj": _shape_tuple(kv_dim, hidden),
        "v_proj": _shape_tuple(kv_dim, hidden),
        "o_proj": _shape_tuple(hidden, q_dim),
        "qkv_proj": _shape_tuple(linear_qkv, hidden),
        "gate": _shape_tuple(intermediate, hidden),
        "up": _shape_tuple(intermediate, hidden),
        "down": _shape_tuple(hidden, intermediate),
    }.get(module, ())
    return _module_shapes_or(by_module, module, fallback, _dominant_dtype(snapshot))


def _shape_for_module(snapshot: ModelSnapshot, module: str) -> str:
    tensor = next((item for item in snapshot.tensors if item.module == module), None)
    if tensor is not None and tensor.shape:
        return f"{shape_text(tensor.shape)} {tensor.dtype}"
    p = snapshot.profile
    hidden = int(p.get("hidden_size") or 0)
    vocab = int(p.get("vocab_size") or 0)
    heads = int(p.get("num_attention_heads") or 0)
    kv_heads = int(p.get("num_key_value_heads") or heads or 0)
    head_dim = int(p.get("head_dim") or (hidden // max(1, heads)) if hidden else 0)
    intermediate = int(p.get("intermediate_size") or 0)
    q_dim = heads * head_dim if heads and head_dim else hidden
    kv_dim = kv_heads * head_dim if kv_heads and head_dim else hidden
    fallback = {
        "embed": _shape_tuple(vocab, hidden),
        "final_norm": _shape_tuple(hidden),
        "lm_head": _shape_tuple(vocab, hidden),
        "q_proj": _shape_tuple(q_dim, hidden),
        "k_proj": _shape_tuple(kv_dim, hidden),
        "v_proj": _shape_tuple(kv_dim, hidden),
        "o_proj": _shape_tuple(hidden, q_dim),
        "gate": _shape_tuple(intermediate, hidden),
        "up": _shape_tuple(intermediate, hidden),
        "down": _shape_tuple(hidden, intermediate),
    }.get(module, ())
    if fallback:
        return f"{shape_text(fallback)} {_dominant_dtype(snapshot)}"
    if tensor is None:
        return "?"
    return f"{shape_text(tensor.shape)} {tensor.dtype}"


def _dominant_dtype(snapshot: ModelSnapshot) -> str:
    counts: Dict[str, int] = defaultdict(int)
    for tensor in snapshot.primary_tensors:
        counts[tensor.dtype] += tensor.numel or 1
    known_counts = {dtype: count for dtype, count in counts.items() if dtype != "unknown"}
    if known_counts:
        return max(known_counts.items(), key=lambda item: item[1])[0]
    profile_dtype = snapshot.profile.get("dtype") or snapshot.profile.get("params_dtype")
    if profile_dtype:
        return normalize_dtype(str(profile_dtype))
    return _config_dtype(snapshot.config)


def _config_dtype(config: Dict[str, object]) -> str:
    value = _recursive_config_value(config, {"dtype", "torch_dtype", "params_dtype"})
    return normalize_dtype(str(value)) if value else "unknown"


def _recursive_config_value(value: object, keys: set) -> object:
    if isinstance(value, dict):
        for key in keys:
            if key in value:
                return value[key]
        for nested in value.values():
            found = _recursive_config_value(nested, keys)
            if found is not None:
                return found
    if isinstance(value, list):
        for item in value:
            found = _recursive_config_value(item, keys)
            if found is not None:
                return found
    return None


def _infer_layer_count(snapshot: ModelSnapshot) -> int:
    layers = [tensor.layer for tensor in snapshot.tensors if tensor.layer is not None]
    return max(layers) + 1 if layers else 0


def _arch_label(profile: Dict[str, object]) -> str:
    layers = int(profile.get("num_hidden_layers") or 0)
    kinds = _profile_layer_kinds(profile, layers) if layers else []
    if kinds and any(_is_linear_attention(kind) for kind in kinds):
        return f"Hybrid: {_schedule_label(profile, kinds).replace('<br/>', ', ')}; {_ffn_label(profile).replace('<br/>', ', ')}"
    return f"{_gqa_label(profile).replace('<br/>', ', ')}; {_ffn_label(profile).replace('<br/>', ', ')}"


def _model_title(snapshot: ModelSnapshot) -> str:
    return f"{snapshot.name}  {_dominant_dtype(snapshot)}  {format_params(snapshot.profile.get('total_params') or snapshot.total_params)}"


def _model_brief(snapshot: ModelSnapshot) -> str:
    p = snapshot.profile
    return f"{_dominant_dtype(snapshot)}, layers={p.get('num_hidden_layers') or _infer_layer_count(snapshot)}, params={format_params(p.get('total_params') or snapshot.total_params)}"


def _bucket_order(name: str) -> int:
    return {
        "Embedding": 0,
        "Attention": 1,
        "MLP / MoE": 2,
        "Norms": 3,
        "Vision Encoder": 4,
        "MTP": 5,
        "LM Head": 6,
        "Quant Aux": 7,
        "KV Cache": 8,
        "State Cache": 9,
        "Other": 10,
    }.get(name, 99)


def _join_sections(sections: Sequence[Tuple[str, str]], output_format: str) -> str:
    if output_format == "html":
        from .formatting import html_page

        text = _join_sections(sections, "markdown")
        return html_page("Model Viewer Report", text)
    chunks: List[str] = []
    for title, body in sections:
        if output_format == "markdown":
            chunks.append(f"## {title}\n\n{_fence_if_needed(body)}")
        else:
            chunks.append(f"== {title} ==\n{body}")
    return "\n\n".join(chunks)


def _fence_if_needed(body: str) -> str:
    stripped = body.strip()
    if stripped.startswith("|") or stripped.startswith("```"):
        return body
    if stripped.startswith("flowchart"):
        return f"```mermaid\n{body}\n```"
    return f"```text\n{body}\n```"


def _esc_mermaid(value: object) -> str:
    return str(value).replace('"', "'").replace("\n", " ")


def _drawio_xml(title: str, nodes: Sequence[Tuple[str, str]]) -> str:
    escaped_title = html.escape(title, quote=True)
    cells = [
        '<mxCell id="0" />',
        '<mxCell id="1" parent="0" />',
    ]
    previous_id = None
    for idx, (label, detail) in enumerate(nodes, start=2):
        cell_id = str(idx)
        x = 60 + (idx - 2) * 230
        value = html.escape(f"{label}\n{detail}".strip(), quote=True)
        cells.append(
            f'<mxCell id="{cell_id}" value="{value}" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="120" width="180" height="72" as="geometry" />'
            "</mxCell>"
        )
        if previous_id is not None:
            edge_id = f"e{idx}"
            cells.append(
                f'<mxCell id="{edge_id}" style="endArrow=block;html=1;rounded=0;" edge="1" parent="1" source="{previous_id}" target="{cell_id}">'
                '<mxGeometry relative="1" as="geometry" />'
                "</mxCell>"
            )
        previous_id = cell_id
    content = "".join(cells)
    return (
        f'<mxfile host="app.diagrams.net"><diagram name="{escaped_title}">'
        f'<mxGraphModel><root>{content}</root></mxGraphModel>'
        "</diagram></mxfile>"
    )


def _pattern_nodes(snapshot: ModelSnapshot, limit: int) -> List[Tuple[str, str]]:
    nodes = []
    for pattern in fold_key_patterns(snapshot)[:limit]:
        detail = f"x{pattern.count}"
        if pattern.shape:
            detail += f" {shape_text(pattern.shape)}"
        if pattern.dtype and pattern.dtype != "unknown":
            detail += f" {pattern.dtype}"
        nodes.append((pattern.pattern, detail))
    return nodes or [("No tensor keys", "")]
