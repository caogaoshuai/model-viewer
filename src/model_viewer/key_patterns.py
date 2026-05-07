from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .schema import ModelSnapshot, TensorInfo


NUMERIC_TOKEN_RE = re.compile(r"^\d+$")


@dataclass(frozen=True)
class KeyPattern:
    pattern: str
    count: int
    shape: Tuple[int, ...]
    dtype: str
    sample_key: str


def fold_key_patterns(snapshot: ModelSnapshot) -> List[KeyPattern]:
    groups: Dict[Tuple[Tuple[str, ...], Tuple[int, ...], str], List[TensorInfo]] = {}
    for tensor in snapshot.tensors:
        parts = tuple(tensor.name.split("."))
        signature = tuple("#" if _is_numeric(part) else part for part in parts)
        key = (signature, tuple(tensor.shape), tensor.dtype)
        groups.setdefault(key, []).append(tensor)

    patterns = []
    for (signature, shape, dtype), tensors in groups.items():
        names = sorted(tensor.name for tensor in tensors)
        pattern = _format_pattern(signature, names)
        patterns.append(
            KeyPattern(
                pattern=pattern,
                count=len(tensors),
                shape=shape,
                dtype=dtype,
                sample_key=names[0],
            )
        )
    return sorted(patterns, key=_pattern_sort_key)


def _format_pattern(signature: Sequence[str], names: Sequence[str]) -> str:
    if len(names) == 1:
        return names[0]

    split_names = [name.split(".") for name in names]
    parts = []
    for idx, token in enumerate(signature):
        if token != "#":
            parts.append(token)
            continue
        values = sorted({int(split_name[idx]) for split_name in split_names})
        parts.append("{" + _format_ranges(values) + "}")
    return ".".join(parts)


def _format_ranges(values: Sequence[int]) -> str:
    if not values:
        return ""
    ranges = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append(_format_range(start, prev))
        start = prev = value
    ranges.append(_format_range(start, prev))
    return ",".join(ranges)


def _format_range(start: int, end: int) -> str:
    if start == end:
        return str(start)
    return f"{start}..{end}"


def _is_numeric(value: str) -> bool:
    return bool(NUMERIC_TOKEN_RE.match(value))


def _pattern_sort_key(pattern: KeyPattern) -> Tuple[int, str]:
    sample_parts = pattern.sample_key.split(".")
    layer = 10**9
    for idx, part in enumerate(sample_parts):
        if (
            part in {"layers", "h", "blocks"}
            and idx + 1 < len(sample_parts)
            and _is_numeric(sample_parts[idx + 1])
        ):
            layer = int(sample_parts[idx + 1])
            break
    return (layer, pattern.sample_key)
