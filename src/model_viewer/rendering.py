from __future__ import annotations

import html
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .diff import MappingRow, ModelDiff
from .formatting import format_bytes, format_params, json_dumps, markdown_table, pct_delta, shape_text
from .schema import ModelSnapshot, TensorInfo, dtype_nbytes


VIEW_ORDER = ("overview", "heatmap", "detail", "mapping", "memory", "tree")
MODULES = ("embed", "ln1", "q_proj", "k_proj", "v_proj", "o_proj", "ln2", "gate", "up", "down")
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
    return _join_sections(sections, output_format)


def render_overview_mermaid(snapshot: ModelSnapshot) -> str:
    profile = snapshot.profile
    title = _model_title(snapshot)
    layers = int(profile.get("num_hidden_layers") or _infer_layer_count(snapshot))
    hidden = profile.get("hidden_size") or "?"
    dtype = _dominant_dtype(snapshot)
    vocab = profile.get("vocab_size") or "?"
    arch = _arch_label(profile)
    lm_head = "LM Head (tied)" if profile.get("tie_word_embeddings") else "LM Head"
    return f"""flowchart TB
    subgraph M["{_esc_mermaid(title)}"]
        EMB["Embedding<br/>[{vocab},{hidden}] {dtype}"]
        DEC["Decoder x {layers}<br/>{_esc_mermaid(arch)}"]
        NORM["Final Norm"]
        LMH["{lm_head}"]
        EMB --> DEC --> NORM --> LMH
    end"""


def render_diff_overview_mermaid(diff: ModelDiff) -> str:
    left = diff.left
    right = diff.right
    summary = diff.summary()
    left_profile = left.profile
    right_profile = right.profile
    left_layers = int(left_profile.get("num_hidden_layers") or _infer_layer_count(left))
    right_layers = int(right_profile.get("num_hidden_layers") or _infer_layer_count(right))
    return f"""flowchart TB
    subgraph LEFT["{_esc_mermaid(_model_title(left))}"]
        L_EMB["Embedding<br/>{_shape_for_module(left, "embed")}"]
        L_DEC["Decoder x {left_layers}<br/>{_esc_mermaid(_arch_label(left_profile))}"]
        L_NORM["Final Norm"]
        L_LMH["LM Head"]
        L_EMB --> L_DEC --> L_NORM --> L_LMH
    end
    subgraph RIGHT["{_esc_mermaid(_model_title(right))}"]
        R_EMB["Embedding<br/>{_shape_for_module(right, "embed")}"]
        R_DEC["Decoder x {right_layers}<br/>{_esc_mermaid(_arch_label(right_profile))}"]
        R_NORM["Final Norm"]
        R_LMH["LM Head{' (tied)' if right_profile.get('tie_word_embeddings') else ''}"]
        R_EMB --> R_DEC --> R_NORM --> R_LMH
    end
    L_DEC -.->|"equiv {summary.get('equivalent', 0)} / diff {summary.get('different', 0) + summary.get('left_only', 0) + summary.get('right_only', 0)}"| R_DEC
    L_LMH -.->|"tied/fused/quant if present"| R_LMH"""


def render_overview_drawio(snapshot: ModelSnapshot) -> str:
    return _drawio_xml(_model_title(snapshot), [
        ("Embedding", _shape_for_module(snapshot, "embed")),
        ("Decoder", f"x {snapshot.profile.get('num_hidden_layers') or _infer_layer_count(snapshot)}"),
        ("Final Norm", ""),
        ("LM Head", "tied" if snapshot.profile.get("tie_word_embeddings") else ""),
    ])


def render_diff_drawio(diff: ModelDiff) -> str:
    summary = diff.summary()
    return _drawio_xml(
        f"{diff.left.name} vs {diff.right.name}",
        [
            (diff.left.name, _model_brief(diff.left)),
            ("Diff", f"exact={summary.get('exact', 0)}, equiv={summary.get('equivalent', 0)}, changed={summary.get('different', 0) + summary.get('left_only', 0) + summary.get('right_only', 0)}"),
            (diff.right.name, _model_brief(diff.right)),
        ],
    )


def render_heatmap(diff: ModelDiff) -> str:
    matrix = _heatmap_matrix(diff)
    layers = sorted(layer for layer in matrix if layer is not None)
    lines = []
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
    return markdown_table(["LEFT key", "RIGHT key", "MATCH", "DETAIL"], rows)


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
    left = dict(_memory_buckets(diff.left, seq_len=seq_len, batch_size=batch_size, include_kv=False))
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
    lines = [
        f"{name} [{format_params(total)} params, {dtype}]",
        f"├── model.embed_tokens [{vocab}, {hidden}]",
        f"├── model.layers.[0..{max(0, layers - 1)}] x {layers}",
        f"│   ├── self_attn q/k/v/o [heads={heads}, kv_heads={kv_heads}]",
        f"│   ├── mlp gate/up/down [intermediate={intermediate}]",
        "│   ├── input_layernorm",
        "│   └── post_attention_layernorm",
        f"├── model.norm [{hidden}]",
    ]
    if p.get("tie_word_embeddings"):
        lines.append("└── lm_head (tied with embedding)")
    else:
        lines.append(f"└── lm_head [{vocab}, {hidden}]")
    if int(p.get("num_experts") or 0):
        lines.insert(4, f"│   ├── moe experts={p.get('num_experts')} active={p.get('num_experts_per_tok')}")
    if int(p.get("num_linear_attn_layers") or 0):
        lines.insert(4, f"│   ├── layer_types linear={p.get('num_linear_attn_layers')} full={p.get('num_standard_attn_layers')}")
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
    lines = [
        f"DECODER BLOCK layer {layer}",
        f"  hidden={snapshot.profile.get('hidden_size', '?')} heads={snapshot.profile.get('num_attention_heads', '?')} kv_heads={snapshot.profile.get('num_key_value_heads', '?')}",
        "  input",
        f"    -> RMSNorm { _module_shapes(by_module, 'ln1') }",
        f"    -> Attention q={_module_shapes(by_module, 'q_proj')} k={_module_shapes(by_module, 'k_proj')} v={_module_shapes(by_module, 'v_proj')} o={_module_shapes(by_module, 'o_proj')}",
        f"    -> MLP gate={_module_shapes(by_module, 'gate')} up={_module_shapes(by_module, 'up')} down={_module_shapes(by_module, 'down')}",
        "    -> residual output",
    ]
    return "\n".join(lines)


def _diff_layer_detail(left: ModelSnapshot, right: ModelSnapshot, layer: int) -> str:
    left_modules = _layer_module_shapes(left, layer)
    right_modules = _layer_module_shapes(right, layer)
    rows = []
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
    buckets: Dict[str, float] = defaultdict(float)
    for tensor in snapshot.tensors:
        module = tensor.module
        if module == "embed":
            bucket = "Embedding"
        elif module in {"q_proj", "k_proj", "v_proj", "qkv_proj", "o_proj"}:
            bucket = "Attention"
        elif module in {"gate", "up", "gate_up", "down"} or tensor.kind in {"expert", "router"}:
            bucket = "MLP / MoE"
        elif module == "lm_head":
            bucket = "LM Head"
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
    return [(name, buckets[name]) for name in sorted(buckets, key=_bucket_order)]


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


def _layer_module_shapes(snapshot: ModelSnapshot, layer: int) -> Dict[str, str]:
    by_module: Dict[str, List[TensorInfo]] = defaultdict(list)
    for tensor in snapshot.tensors:
        if tensor.layer == layer:
            by_module[tensor.module].append(tensor)
    return {module: _module_shapes(by_module, module) for module in by_module}


def _shape_for_module(snapshot: ModelSnapshot, module: str) -> str:
    tensor = next((item for item in snapshot.tensors if item.module == module), None)
    if tensor is None:
        return "?"
    return f"{shape_text(tensor.shape)} {tensor.dtype}"


def _dominant_dtype(snapshot: ModelSnapshot) -> str:
    counts: Dict[str, int] = defaultdict(int)
    for tensor in snapshot.primary_tensors:
        counts[tensor.dtype] += tensor.numel or 1
    if not counts:
        return "unknown"
    return max(counts.items(), key=lambda item: item[1])[0]


def _infer_layer_count(snapshot: ModelSnapshot) -> int:
    layers = [tensor.layer for tensor in snapshot.tensors if tensor.layer is not None]
    return max(layers) + 1 if layers else 0


def _arch_label(profile: Dict[str, object]) -> str:
    parts = []
    if int(profile.get("num_experts") or 0):
        parts.append(f"MoE experts={profile.get('num_experts')}")
    else:
        parts.append("q/k/v/o + gate/up/down")
    if int(profile.get("num_linear_attn_layers") or 0):
        parts.append(f"linear-attn={profile.get('num_linear_attn_layers')}")
    return ", ".join(parts)


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
        "LM Head": 4,
        "Quant Aux": 5,
        "KV Cache": 6,
        "Other": 7,
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
