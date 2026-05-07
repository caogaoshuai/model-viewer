from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


SNAPSHOT_SCHEMA = "model-viewer.snapshot.v1"


DTYPE_BYTES = {
    "f64": 8.0,
    "float64": 8.0,
    "f32": 4.0,
    "fp32": 4.0,
    "float32": 4.0,
    "tf32": 4.0,
    "f16": 2.0,
    "fp16": 2.0,
    "float16": 2.0,
    "bf16": 2.0,
    "bfloat16": 2.0,
    "f8_e4m3": 1.0,
    "f8_e5m2": 1.0,
    "fp8": 1.0,
    "i64": 8.0,
    "int64": 8.0,
    "i32": 4.0,
    "int32": 4.0,
    "u32": 4.0,
    "i16": 2.0,
    "int16": 2.0,
    "u16": 2.0,
    "i8": 1.0,
    "int8": 1.0,
    "u8": 1.0,
    "uint8": 1.0,
    "bool": 1.0,
    "int4": 0.5,
    "uint4": 0.5,
}


SAFETENSORS_DTYPE_ALIASES = {
    "F64": "f64",
    "F32": "f32",
    "F16": "fp16",
    "BF16": "bf16",
    "F8_E4M3": "f8_e4m3",
    "F8_E5M2": "f8_e5m2",
    "I64": "i64",
    "I32": "i32",
    "I16": "i16",
    "I8": "i8",
    "U64": "u64",
    "U32": "u32",
    "U16": "u16",
    "U8": "u8",
    "BOOL": "bool",
}


def normalize_dtype(dtype: Optional[str]) -> str:
    if not dtype:
        return "unknown"
    value = str(dtype).strip()
    if value.startswith("torch."):
        value = value[len("torch.") :]
    upper = value.upper()
    if upper in SAFETENSORS_DTYPE_ALIASES:
        return SAFETENSORS_DTYPE_ALIASES[upper]
    aliases = {
        "bfloat16": "bf16",
        "float16": "fp16",
        "half": "fp16",
        "float32": "fp32",
        "float": "fp32",
        "float64": "f64",
        "double": "f64",
        "float8_e4m3fn": "f8_e4m3",
        "float8_e5m2": "f8_e5m2",
    }
    lowered = value.lower()
    return aliases.get(lowered, lowered)


def dtype_nbytes(dtype: Optional[str]) -> float:
    return DTYPE_BYTES.get(normalize_dtype(dtype), 0.0)


def prod(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


@dataclass
class TensorInfo:
    name: str
    shape: Tuple[int, ...] = field(default_factory=tuple)
    dtype: str = "unknown"
    byte_size: Optional[int] = None
    kind: str = "weight"
    layer: Optional[int] = None
    module: str = ""
    canonical_name: Optional[str] = None
    parent: Optional[str] = None
    source: str = "unknown"

    @property
    def numel(self) -> int:
        if not self.shape:
            return 0
        return prod(self.shape)

    @property
    def logical_bytes(self) -> float:
        if self.byte_size is not None:
            return float(self.byte_size)
        return self.numel * dtype_nbytes(self.dtype)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "byte_size": self.byte_size,
            "kind": self.kind,
            "layer": self.layer,
            "module": self.module,
            "canonical_name": self.canonical_name,
            "parent": self.parent,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TensorInfo":
        return cls(
            name=str(raw["name"]),
            shape=tuple(int(v) for v in raw.get("shape", []) if v is not None),
            dtype=normalize_dtype(raw.get("dtype")),
            byte_size=raw.get("byte_size"),
            kind=str(raw.get("kind") or "weight"),
            layer=raw.get("layer"),
            module=str(raw.get("module") or ""),
            canonical_name=raw.get("canonical_name"),
            parent=raw.get("parent"),
            source=str(raw.get("source") or "unknown"),
        )


@dataclass
class ModelSnapshot:
    name: str
    source: str
    tensors: List[TensorInfo] = field(default_factory=list)
    profile: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    config_path: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    schema_version: str = SNAPSHOT_SCHEMA

    @property
    def primary_tensors(self) -> List[TensorInfo]:
        return [tensor for tensor in self.tensors if tensor.kind != "quant_aux"]

    @property
    def total_params(self) -> int:
        return sum(tensor.numel for tensor in self.primary_tensors)

    @property
    def total_bytes(self) -> float:
        return sum(tensor.logical_bytes for tensor in self.tensors)

    def tensor_by_name(self) -> Dict[str, TensorInfo]:
        return {tensor.name: tensor for tensor in self.tensors}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "source": self.source,
            "config_path": self.config_path,
            "config": self.config,
            "profile": self.profile,
            "warnings": self.warnings,
            "tensors": [tensor.to_dict() for tensor in self.tensors],
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ModelSnapshot":
        if raw.get("schema_version") != SNAPSHOT_SCHEMA:
            raise ValueError("Not a model-viewer snapshot JSON.")
        return cls(
            name=str(raw.get("name") or "snapshot"),
            source=str(raw.get("source") or "snapshot"),
            config_path=raw.get("config_path"),
            config=dict(raw.get("config") or {}),
            profile=dict(raw.get("profile") or {}),
            warnings=list(raw.get("warnings") or []),
            tensors=[TensorInfo.from_dict(item) for item in raw.get("tensors", [])],
        )
