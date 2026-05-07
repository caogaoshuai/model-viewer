from __future__ import annotations

import importlib.util
import json
import os
import re
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .schema import ModelSnapshot, SNAPSHOT_SCHEMA, TensorInfo, dtype_nbytes, normalize_dtype


MODEL_CONFIG_KEYS = {"language_config", "llm_config", "text_config", "model_config"}
QUANT_AUX_SUFFIXES = (
    ".scales",
    ".zeros",
    ".qzeros",
    ".g_idx",
    ".gidx",
    ".weight_scale",
    ".weight_scale_inv",
    ".input_scale",
    ".activation_scale",
)
QUANT_WEIGHT_SUFFIXES = (".qweight", ".packed_weight")
LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)(?:\.|$)")


def load_model(
    ref: str,
    model_source: str = "auto",
    revision: Optional[str] = None,
    hub_token: Optional[str] = None,
) -> ModelSnapshot:
    path = Path(ref).expanduser()
    if path.exists():
        return _load_local(path)
    return _load_remote(ref, model_source=model_source, revision=revision, hub_token=hub_token)


def _load_local(path: Path) -> ModelSnapshot:
    if path.is_file() and path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if raw.get("schema_version") == SNAPSHOT_SCHEMA:
            return ModelSnapshot.from_dict(raw)
        if path.name.endswith(".safetensors.index.json"):
            model_dir = path.parent
            return _snapshot_from_parts(
                name=model_dir.name or path.stem,
                source=str(path),
                config_path=_optional_config(model_dir),
                tensor_sources=[path],
            )
        return _snapshot_from_parts(
            name=path.parent.name or path.stem,
            source=str(path),
            config_path=path,
            tensor_sources=[],
        )

    if path.is_file() and path.suffix == ".safetensors":
        return _snapshot_from_parts(
            name=path.stem,
            source=str(path),
            config_path=_optional_config(path.parent),
            tensor_sources=[path],
        )

    if path.is_dir():
        config_path = _optional_config(path)
        tensor_sources = _find_tensor_sources(path)
        return _snapshot_from_parts(
            name=path.name,
            source=str(path),
            config_path=config_path,
            tensor_sources=tensor_sources,
        )

    raise FileNotFoundError(f"Unsupported model input: {path}")


def _load_remote(
    ref: str,
    model_source: str = "auto",
    revision: Optional[str] = None,
    hub_token: Optional[str] = None,
) -> ModelSnapshot:
    normalized, prefix_source = _normalize_remote_ref(ref)
    source = prefix_source if model_source == "auto" and prefix_source != "auto" else model_source
    attempts: List[str]
    if source == "hf":
        attempts = ["hf"]
    elif source == "ms":
        attempts = ["ms"]
    else:
        attempts = ["hf", "ms"]

    errors: List[str] = []
    for attempt in attempts:
        try:
            if attempt == "hf":
                directory = _download_hf_metadata(normalized, revision, hub_token)
            else:
                directory = _download_ms_metadata(normalized, revision, hub_token)
            snapshot = _load_local(Path(directory))
            snapshot.name = normalized
            snapshot.source = f"{attempt}://{normalized}"
            return snapshot
        except Exception as error:  # pragma: no cover - depends on optional hub packages.
            errors.append(f"{attempt}: {error}")
    raise RuntimeError(f"Failed to resolve remote model {ref!r}: {'; '.join(errors)}")


def _normalize_remote_ref(ref: str) -> Tuple[str, str]:
    if ref.startswith("hf://"):
        return ref[len("hf://") :], "hf"
    if ref.startswith("ms://"):
        return ref[len("ms://") :], "ms"
    if ref.startswith("modelscope/"):
        return ref[len("modelscope/") :], "ms"
    return ref, "auto"


def _download_hf_metadata(ref: str, revision: Optional[str], hub_token: Optional[str]) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - optional dependency.
        raise RuntimeError("huggingface_hub is not installed. Install model-viewer[hub].") from error
    return snapshot_download(
        ref,
        repo_type="model",
        revision=revision,
        token=hub_token,
        allow_patterns=["config.json", "*.safetensors.index.json"],
    )


def _download_ms_metadata(ref: str, revision: Optional[str], hub_token: Optional[str]) -> str:
    try:
        from modelscope import snapshot_download
    except ImportError as error:  # pragma: no cover - optional dependency.
        raise RuntimeError("modelscope is not installed. Install model-viewer[hub].") from error
    kwargs: Dict[str, Any] = {"allow_patterns": ["config.json", "*.safetensors.index.json"]}
    if hub_token:
        kwargs["token"] = hub_token
    return snapshot_download(ref, revision=revision or "master", **kwargs)


def _snapshot_from_parts(
    name: str,
    source: str,
    config_path: Optional[Path],
    tensor_sources: Sequence[Path],
) -> ModelSnapshot:
    warnings: List[str] = []
    config: Dict[str, Any] = {}
    profile: Dict[str, Any] = {}
    if config_path is not None:
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
            profile = _load_profile(config_path, config)
        except Exception as error:
            warnings.append(f"Failed to parse config {config_path}: {error}")

    tensors: List[TensorInfo] = []
    for tensor_source in tensor_sources:
        try:
            tensors.extend(_read_tensor_source(tensor_source))
        except Exception as error:
            warnings.append(f"Failed to parse tensor metadata {tensor_source}: {error}")

    if not tensors and profile:
        tensors = _synthesize_tensors(profile, config)
        warnings.append("No tensor metadata found; synthesized a structural tensor list from config.json.")

    tensors = [_classify_tensor(tensor) for tensor in tensors]
    return ModelSnapshot(
        name=name,
        source=source,
        config_path=str(config_path) if config_path else None,
        config=config,
        profile=profile,
        tensors=sorted(tensors, key=_tensor_sort_key),
        warnings=warnings,
    )


def _optional_config(directory: Path) -> Optional[Path]:
    candidate = directory / "config.json"
    return candidate if candidate.is_file() else None


def _find_tensor_sources(directory: Path) -> List[Path]:
    indexes = sorted(directory.glob("*.safetensors.index.json"))
    if indexes:
        return indexes
    return sorted(directory.glob("*.safetensors"))


def _read_tensor_source(path: Path) -> List[TensorInfo]:
    if path.name.endswith(".safetensors.index.json"):
        return _read_safetensors_index(path)
    return _read_safetensors_file(path)


def _read_safetensors_index(path: Path) -> List[TensorInfo]:
    with path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map") or {}
    if not isinstance(weight_map, dict):
        raise ValueError("safetensors index has no weight_map object")

    tensors: Dict[str, TensorInfo] = {}
    by_file: Dict[str, List[str]] = {}
    for key, filename in weight_map.items():
        by_file.setdefault(str(filename), []).append(str(key))

    for filename, keys in by_file.items():
        shard = path.parent / filename
        if not shard.is_file():
            for key in keys:
                tensors[key] = TensorInfo(name=key, source="safetensors-index")
            continue
        for tensor in _read_safetensors_file(shard):
            if tensor.name in keys:
                tensors[tensor.name] = tensor

    return list(tensors.values())


def _read_safetensors_file(path: Path) -> List[TensorInfo]:
    with path.open("rb") as handle:
        header_len_raw = handle.read(8)
        if len(header_len_raw) != 8:
            raise ValueError("invalid safetensors header")
        header_len = struct.unpack("<Q", header_len_raw)[0]
        header = json.loads(handle.read(header_len))

    tensors: List[TensorInfo] = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        shape = tuple(int(value) for value in meta.get("shape") or [])
        dtype = normalize_dtype(meta.get("dtype"))
        offsets = meta.get("data_offsets") or []
        byte_size = None
        if len(offsets) == 2:
            byte_size = int(offsets[1]) - int(offsets[0])
        tensors.append(
            TensorInfo(
                name=str(name),
                shape=shape,
                dtype=dtype,
                byte_size=byte_size,
                source="safetensors",
            )
        )
    return tensors


def _load_profile(config_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    estimator_profile = _try_training_resource_estimator(config_path)
    if estimator_profile is not None:
        estimator_profile["profile_source"] = "training-resource-estimator"
        return estimator_profile
    profile = _fallback_profile(config)
    profile["profile_source"] = "model-viewer"
    return profile


def _try_training_resource_estimator(config_path: Path) -> Optional[Dict[str, Any]]:
    candidates = []
    env_dir = os.environ.get("TRAINING_RESOURCE_ESTIMATOR_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(Path(__file__).resolve().parents[3] / "training-resource-estimator")
    candidates.append(Path("/Users/cgs/Documents/project/training-resource-estimator"))

    for root in candidates:
        trainer_core = root / "frameworks" / "swift" / "deepspeed" / "core" / "trainer_core.py"
        if not trainer_core.is_file():
            continue
        spec = importlib.util.spec_from_file_location("_mad_training_resource_estimator", trainer_core)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            profile = module.load_profile_from_config(str(config_path))
            if isinstance(profile, dict):
                return _jsonable(profile)
        except Exception:
            continue
    return None


def _fallback_profile(config: Dict[str, Any]) -> Dict[str, Any]:
    hidden = _first_config_attr(config, ("hidden_size", "n_embd", "d_model")) or 0
    layers = _first_config_attr(config, ("num_hidden_layers", "n_layer", "num_layers")) or 0
    heads = _first_config_attr(config, ("num_attention_heads", "n_head", "encoder_attention_heads")) or 0
    kv_heads = _first_config_attr(config, ("num_key_value_heads",)) or heads
    head_dim = _first_config_attr(config, ("head_dim",)) or (int(hidden) // max(1, int(heads or 1)))
    vocab = _first_config_attr(config, ("vocab_size",)) or 0
    intermediate = _first_config_attr(config, ("intermediate_size", "n_inner"))
    moe_intermediate = _first_config_attr(config, ("moe_intermediate_size",))
    if intermediate is None:
        intermediate = moe_intermediate or int(hidden) * 4
    layer_types = _get_config_attr(config, "layer_types")
    tie = bool(_get_config_attr(config, "tie_word_embeddings") or False)
    max_len = _first_config_attr(config, ("max_position_embeddings", "seq_length", "max_seq_len"))
    num_experts = _first_config_attr(config, ("num_experts",)) or 0
    experts_per_tok = _first_config_attr(config, ("num_experts_per_tok",)) or 0
    shared = _first_config_attr(config, ("shared_expert_intermediate_size",)) or 0
    model_type = str(config.get("model_type") or "").lower()
    kinds = _layer_kinds(int(layers or 0), layer_types)
    total = _estimate_total_params(
        hidden=int(hidden or 0),
        intermediate=int(intermediate or 0),
        layers=int(layers or 0),
        heads=int(heads or 0),
        kv_heads=int(kv_heads or 0),
        head_dim=int(head_dim or 0),
        vocab=int(vocab or 0),
        tie=tie,
        num_experts=int(num_experts or 0),
        moe_intermediate=int(moe_intermediate or 0),
        shared_expert_intermediate=int(shared or 0),
    )
    return {
        "model_type": model_type,
        "architecture_type": "moe" if num_experts else "dense",
        "total_params": total,
        "hidden_size": int(hidden or 0),
        "intermediate_size": int(intermediate or 0),
        "num_hidden_layers": int(layers or 0),
        "num_attention_heads": int(heads or 0),
        "num_key_value_heads": int(kv_heads or 0),
        "head_dim": int(head_dim or 0),
        "vocab_size": int(vocab or 0),
        "tie_word_embeddings": tie,
        "default_max_model_len": int(max_len) if max_len else None,
        "num_experts": int(num_experts or 0),
        "num_experts_per_tok": int(experts_per_tok or 0),
        "moe_intermediate_size": int(moe_intermediate or 0),
        "shared_expert_intermediate_size": int(shared or 0),
        "num_moe_layers": int(layers or 0) if num_experts else 0,
        "num_dense_layers": 0 if num_experts else int(layers or 0),
        "num_linear_attn_layers": sum(1 for kind in kinds if kind == "linear_attention"),
        "num_standard_attn_layers": sum(1 for kind in kinds if kind != "linear_attention"),
        "layer_kinds": kinds,
    }


def _estimate_total_params(
    hidden: int,
    intermediate: int,
    layers: int,
    heads: int,
    kv_heads: int,
    head_dim: int,
    vocab: int,
    tie: bool,
    num_experts: int,
    moe_intermediate: int,
    shared_expert_intermediate: int,
) -> int:
    q_dim = heads * head_dim
    kv_dim = kv_heads * head_dim
    attn = hidden * (q_dim + 2 * kv_dim) + q_dim * hidden
    norm = 4 * hidden
    if num_experts and moe_intermediate:
        expert = num_experts * 3 * hidden * moe_intermediate
        router = hidden * num_experts
        shared = 3 * hidden * shared_expert_intermediate
        layer_params = attn + norm + expert + router + shared
    else:
        layer_params = attn + norm + 3 * hidden * intermediate
    return vocab * hidden + layers * layer_params + hidden + (0 if tie else vocab * hidden)


def _first_config_attr(config: Dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        value = _get_config_attr(config, name)
        if value is not None:
            return value
    return None


def _get_config_attr(config: Any, attr_name: str, parent_key: Optional[str] = None) -> Any:
    if isinstance(config, dict):
        allowed_parent = parent_key is None or parent_key in MODEL_CONFIG_KEYS
        if allowed_parent and attr_name in config:
            return config[attr_name]
        for key, value in config.items():
            if key.endswith("_config") or key in MODEL_CONFIG_KEYS:
                nested = _get_config_attr(value, attr_name, key)
                if nested is not None:
                    return nested
    elif isinstance(config, list):
        for item in config:
            value = _get_config_attr(item, attr_name, parent_key)
            if value is not None:
                return value
    return None


def _layer_kinds(num_layers: int, layer_types: Any) -> List[str]:
    if isinstance(layer_types, list) and layer_types:
        kinds = [
            "linear_attention" if ("linear" in str(item).lower() or "delta" in str(item).lower()) else "full_attention"
            for item in layer_types
        ]
        if len(kinds) < num_layers:
            kinds.extend(["full_attention"] * (num_layers - len(kinds)))
        return kinds[:num_layers]
    return ["full_attention"] * num_layers


def _synthesize_tensors(profile: Dict[str, Any], config: Dict[str, Any]) -> List[TensorInfo]:
    hidden = int(profile.get("hidden_size") or 0)
    layers = int(profile.get("num_hidden_layers") or 0)
    heads = int(profile.get("num_attention_heads") or 0)
    kv_heads = int(profile.get("num_key_value_heads") or heads)
    head_dim = int(profile.get("head_dim") or (hidden // max(1, heads)))
    q_dim = heads * head_dim
    kv_dim = kv_heads * head_dim
    vocab = int(profile.get("vocab_size") or 0)
    intermediate = int(profile.get("intermediate_size") or 0)
    tie = bool(profile.get("tie_word_embeddings"))
    dtype = _infer_config_dtype(config)
    linear_dtype = _infer_linear_dtype(config, dtype)

    tensors = [
        _tensor("model.embed_tokens.weight", (vocab, hidden), dtype, "synthetic"),
    ]
    for idx in range(layers):
        prefix = f"model.layers.{idx}"
        tensors.extend(
            [
                _tensor(f"{prefix}.input_layernorm.weight", (hidden,), dtype, "synthetic"),
                _tensor(f"{prefix}.self_attn.q_proj.weight", (q_dim, hidden), linear_dtype, "synthetic"),
                _tensor(f"{prefix}.self_attn.k_proj.weight", (kv_dim, hidden), linear_dtype, "synthetic"),
                _tensor(f"{prefix}.self_attn.v_proj.weight", (kv_dim, hidden), linear_dtype, "synthetic"),
                _tensor(f"{prefix}.self_attn.o_proj.weight", (hidden, q_dim), linear_dtype, "synthetic"),
                _tensor(f"{prefix}.post_attention_layernorm.weight", (hidden,), dtype, "synthetic"),
            ]
        )
        if int(profile.get("num_experts") or 0) and int(profile.get("moe_intermediate_size") or 0):
            experts = int(profile.get("num_experts") or 0)
            moe_i = int(profile.get("moe_intermediate_size") or 0)
            tensors.append(_tensor(f"{prefix}.mlp.gate.weight", (experts, hidden), dtype, "synthetic"))
            for expert_idx in range(experts):
                expert = f"{prefix}.mlp.experts.{expert_idx}"
                tensors.extend(
                    [
                        _tensor(f"{expert}.gate_proj.weight", (moe_i, hidden), linear_dtype, "synthetic"),
                        _tensor(f"{expert}.up_proj.weight", (moe_i, hidden), linear_dtype, "synthetic"),
                        _tensor(f"{expert}.down_proj.weight", (hidden, moe_i), linear_dtype, "synthetic"),
                    ]
                )
        else:
            tensors.extend(
                [
                    _tensor(f"{prefix}.mlp.gate_proj.weight", (intermediate, hidden), linear_dtype, "synthetic"),
                    _tensor(f"{prefix}.mlp.up_proj.weight", (intermediate, hidden), linear_dtype, "synthetic"),
                    _tensor(f"{prefix}.mlp.down_proj.weight", (hidden, intermediate), linear_dtype, "synthetic"),
                ]
            )
    tensors.append(_tensor("model.norm.weight", (hidden,), dtype, "synthetic"))
    if not tie:
        tensors.append(_tensor("lm_head.weight", (vocab, hidden), dtype, "synthetic"))
    return tensors


def _tensor(name: str, shape: Tuple[int, ...], dtype: str, source: str) -> TensorInfo:
    return TensorInfo(name=name, shape=shape, dtype=normalize_dtype(dtype), source=source)


def _infer_config_dtype(config: Dict[str, Any]) -> str:
    value = _first_config_attr(config, ("torch_dtype", "dtype", "params_dtype"))
    return normalize_dtype(value or "bf16")


def _infer_linear_dtype(config: Dict[str, Any], fallback: str) -> str:
    quant = _get_config_attr(config, "quantization_config")
    if isinstance(quant, dict):
        bits = quant.get("bits")
        if bits in (4, "4"):
            return "int4"
        if bits in (8, "8"):
            return "int8"
        method = str(quant.get("quant_method") or "").lower()
        if "fp8" in method:
            return "fp8"
    return fallback


def _classify_tensor(tensor: TensorInfo) -> TensorInfo:
    name = tensor.name
    canonical = _canonical_name(name)
    tensor.canonical_name = canonical
    tensor.parent = _quant_parent(name)
    tensor.layer = _layer_index(name)
    tensor.module = _module_name(canonical)
    if tensor.parent:
        tensor.kind = "quant_aux"
    elif "embed_tokens" in name or "wte" in name:
        tensor.kind = "embedding"
    elif name.startswith("lm_head") or ".lm_head" in name:
        tensor.kind = "lm_head"
    elif "norm" in name or "ln_" in name or "layernorm" in name:
        tensor.kind = "norm"
    elif "expert" in name:
        tensor.kind = "expert"
    elif ".gate." in name and ".mlp." in name:
        tensor.kind = "router"
    else:
        tensor.kind = "weight"
    tensor.dtype = normalize_dtype(tensor.dtype)
    if tensor.byte_size is None and tensor.dtype == "unknown" and tensor.shape:
        tensor.byte_size = int(tensor.numel * dtype_nbytes("bf16"))
    return tensor


def _canonical_name(name: str) -> str:
    canonical = name
    for suffix in QUANT_WEIGHT_SUFFIXES:
        if canonical.endswith(suffix):
            canonical = canonical[: -len(suffix)] + ".weight"
            break
    for suffix in QUANT_AUX_SUFFIXES:
        if canonical.endswith(suffix):
            canonical = canonical[: -len(suffix)] + ".weight"
            break
    return canonical


def _quant_parent(name: str) -> Optional[str]:
    for suffix in QUANT_AUX_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)] + ".weight"
    return None


def _layer_index(name: str) -> Optional[int]:
    match = LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def _module_name(canonical_name: str) -> str:
    aliases = {
        "embed_tokens": "embed",
        "wte": "embed",
        "input_layernorm": "ln1",
        "ln_1": "ln1",
        "q_proj": "q_proj",
        "k_proj": "k_proj",
        "v_proj": "v_proj",
        "qkv_proj": "qkv_proj",
        "query_key_value": "qkv_proj",
        "c_attn": "qkv_proj",
        "o_proj": "o_proj",
        "out_proj": "o_proj",
        "post_attention_layernorm": "ln2",
        "ln_2": "ln2",
        "gate_proj": "gate",
        "up_proj": "up",
        "gate_up_proj": "gate_up",
        "down_proj": "down",
        "lm_head": "lm_head",
        "norm": "final_norm",
    }
    parts = canonical_name.split(".")
    for part in reversed(parts):
        if part in aliases:
            return aliases[part]
    if "embed_tokens" in canonical_name:
        return "embed"
    if canonical_name.startswith("lm_head"):
        return "lm_head"
    return parts[-2] if len(parts) >= 2 and parts[-1] == "weight" else parts[-1]


def _tensor_sort_key(tensor: TensorInfo) -> Tuple[int, int, str]:
    layer = tensor.layer if tensor.layer is not None else -1
    module_order = {
        "embed": 0,
        "ln1": 1,
        "q_proj": 2,
        "k_proj": 3,
        "v_proj": 4,
        "qkv_proj": 5,
        "o_proj": 6,
        "ln2": 7,
        "gate": 8,
        "up": 9,
        "gate_up": 10,
        "down": 11,
        "final_norm": 12,
        "lm_head": 13,
    }.get(tensor.module, 50)
    return (layer, module_order, tensor.name)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
