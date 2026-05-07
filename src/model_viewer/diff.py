from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .schema import ModelSnapshot, TensorInfo


@dataclass
class MappingRow:
    left: List[TensorInfo] = field(default_factory=list)
    right: List[TensorInfo] = field(default_factory=list)
    status: str = "exact"
    reason: str = ""
    layer: Optional[int] = None
    modules: List[str] = field(default_factory=list)

    @property
    def left_names(self) -> List[str]:
        return [tensor.name for tensor in self.left]

    @property
    def right_names(self) -> List[str]:
        return [tensor.name for tensor in self.right]


@dataclass
class ModelDiff:
    left: ModelSnapshot
    right: ModelSnapshot
    rows: List[MappingRow]
    fuzzy_match: bool = False
    ignore_quantization: bool = False

    @property
    def has_change(self) -> bool:
        return any(row.status != "exact" for row in self.rows)

    def summary(self) -> Dict[str, int]:
        counts = {
            "exact": 0,
            "equivalent": 0,
            "different": 0,
            "left_only": 0,
            "right_only": 0,
            "auxiliary": 0,
        }
        for row in self.rows:
            counts[row.status] = counts.get(row.status, 0) + 1
        return counts


def compare_models(
    left: ModelSnapshot,
    right: ModelSnapshot,
    fuzzy_match: bool = False,
    ignore_quantization: bool = False,
) -> ModelDiff:
    left_primary = _primary_by_name(left)
    right_primary = _primary_by_name(right)
    unmatched_left: Set[str] = set(left_primary)
    unmatched_right: Set[str] = set(right_primary)
    rows: List[MappingRow] = []

    for name in sorted(unmatched_left & unmatched_right):
        left_tensor = left_primary[name]
        right_tensor = right_primary[name]
        rows.append(
            _compare_row(
                [left_tensor],
                [right_tensor],
                "exact key",
                ignore_quantization=ignore_quantization,
            )
        )
        unmatched_left.remove(name)
        unmatched_right.remove(name)

    _match_canonical(
        left_primary,
        right_primary,
        unmatched_left,
        unmatched_right,
        rows,
        ignore_quantization=ignore_quantization,
    )

    if fuzzy_match:
        _match_fused_qkv(
            left_primary,
            right_primary,
            unmatched_left,
            unmatched_right,
            rows,
            ignore_quantization=ignore_quantization,
        )
        _match_fused_gate_up(
            left_primary,
            right_primary,
            unmatched_left,
            unmatched_right,
            rows,
            ignore_quantization=ignore_quantization,
        )
        _match_tied_lm_head(left_primary, right_primary, unmatched_left, unmatched_right, rows)

    if not ignore_quantization:
        _append_auxiliary_rows(left, right, rows)

    for name in sorted(unmatched_left):
        tensor = left_primary[name]
        rows.append(
            MappingRow(
                left=[tensor],
                status="left_only",
                reason="missing on right",
                layer=tensor.layer,
                modules=[tensor.module],
            )
        )
    for name in sorted(unmatched_right):
        tensor = right_primary[name]
        rows.append(
            MappingRow(
                right=[tensor],
                status="right_only",
                reason="new on right",
                layer=tensor.layer,
                modules=[tensor.module],
            )
        )

    return ModelDiff(
        left=left,
        right=right,
        rows=sorted(rows, key=_row_sort_key),
        fuzzy_match=fuzzy_match,
        ignore_quantization=ignore_quantization,
    )


def _primary_by_name(snapshot: ModelSnapshot) -> Dict[str, TensorInfo]:
    return {tensor.name: tensor for tensor in snapshot.tensors if tensor.kind != "quant_aux"}


def _match_canonical(
    left_primary: Dict[str, TensorInfo],
    right_primary: Dict[str, TensorInfo],
    unmatched_left: Set[str],
    unmatched_right: Set[str],
    rows: List[MappingRow],
    ignore_quantization: bool = False,
) -> None:
    right_by_canonical: Dict[str, str] = {}
    for name in unmatched_right:
        canonical = right_primary[name].canonical_name or name
        right_by_canonical.setdefault(canonical, name)

    for left_name in sorted(list(unmatched_left)):
        left_tensor = left_primary[left_name]
        canonical = left_tensor.canonical_name or left_name
        right_name = right_by_canonical.get(canonical)
        if right_name is None or right_name not in unmatched_right:
            continue
        right_tensor = right_primary[right_name]
        rows.append(
            _compare_row(
                [left_tensor],
                [right_tensor],
                "canonical key",
                ignore_quantization=ignore_quantization,
            )
        )
        unmatched_left.remove(left_name)
        unmatched_right.remove(right_name)


def _match_fused_qkv(
    left_primary: Dict[str, TensorInfo],
    right_primary: Dict[str, TensorInfo],
    unmatched_left: Set[str],
    unmatched_right: Set[str],
    rows: List[MappingRow],
    ignore_quantization: bool = False,
) -> None:
    left_by_layer = _by_layer_module(left_primary, unmatched_left)
    right_by_layer = _by_layer_module(right_primary, unmatched_right)
    for layer, left_modules in sorted(left_by_layer.items()):
        qkv_names = [
            _first_name(left_modules.get(module, []))
            for module in ("q_proj", "k_proj", "v_proj")
        ]
        if any(name is None for name in qkv_names):
            continue
        right_name = _first_name(right_by_layer.get(layer, {}).get("qkv_proj", []))
        if right_name is None:
            continue
        left_group = [left_primary[name] for name in qkv_names if name is not None]
        right_tensor = right_primary[right_name]
        rows.append(
            _compare_fused_row(
                left_group,
                [right_tensor],
                "fused q+k+v",
                ignore_quantization=ignore_quantization,
            )
        )
        for name in qkv_names:
            unmatched_left.remove(name)
        unmatched_right.remove(right_name)


def _match_fused_gate_up(
    left_primary: Dict[str, TensorInfo],
    right_primary: Dict[str, TensorInfo],
    unmatched_left: Set[str],
    unmatched_right: Set[str],
    rows: List[MappingRow],
    ignore_quantization: bool = False,
) -> None:
    left_by_layer = _by_layer_module(left_primary, unmatched_left)
    right_by_layer = _by_layer_module(right_primary, unmatched_right)
    for layer, left_modules in sorted(left_by_layer.items()):
        names = [
            _first_name(left_modules.get(module, []))
            for module in ("gate", "up")
        ]
        if any(name is None for name in names):
            continue
        right_name = _first_name(right_by_layer.get(layer, {}).get("gate_up", []))
        if right_name is None:
            continue
        left_group = [left_primary[name] for name in names if name is not None]
        right_tensor = right_primary[right_name]
        rows.append(
            _compare_fused_row(
                left_group,
                [right_tensor],
                "fused gate+up",
                ignore_quantization=ignore_quantization,
            )
        )
        for name in names:
            unmatched_left.remove(name)
        unmatched_right.remove(right_name)


def _match_tied_lm_head(
    left_primary: Dict[str, TensorInfo],
    right_primary: Dict[str, TensorInfo],
    unmatched_left: Set[str],
    unmatched_right: Set[str],
    rows: List[MappingRow],
) -> None:
    left_lm = [name for name in unmatched_left if left_primary[name].module == "lm_head"]
    if not left_lm:
        return
    right_embed = next((tensor for tensor in right_primary.values() if tensor.module == "embed"), None)
    if right_embed is None:
        return
    for name in left_lm:
        left_tensor = left_primary[name]
        if _shape_compatible([left_tensor], [right_embed]):
            rows.append(
                MappingRow(
                    left=[left_tensor],
                    right=[right_embed],
                    status="equivalent",
                    reason="tied with embedding",
                    layer=None,
                    modules=["lm_head"],
                )
            )
            unmatched_left.remove(name)


def _append_auxiliary_rows(left: ModelSnapshot, right: ModelSnapshot, rows: List[MappingRow]) -> None:
    for tensor in left.tensors:
        if tensor.kind == "quant_aux":
            rows.append(
                MappingRow(
                    left=[tensor],
                    status="auxiliary",
                    reason="left quantization auxiliary tensor",
                    layer=tensor.layer,
                    modules=[tensor.module],
                )
            )
    for tensor in right.tensors:
        if tensor.kind == "quant_aux":
            rows.append(
                MappingRow(
                    right=[tensor],
                    status="auxiliary",
                    reason="right quantization auxiliary tensor",
                    layer=tensor.layer,
                    modules=[tensor.module],
                )
            )


def _compare_row(
    left: List[TensorInfo],
    right: List[TensorInfo],
    reason: str,
    ignore_quantization: bool = False,
) -> MappingRow:
    status = "exact"
    detail = reason
    quantization_ignored = ignore_quantization and _is_quantization_related(left, right)
    if quantization_ignored:
        detail = f"{reason}; quantization ignored"
    elif not _shape_equal(left, right):
        status = "different"
        detail = f"{reason}; shape mismatch"
    elif not _dtype_equal(left, right):
        status = "equivalent"
        detail = f"{reason}; dtype differs"
    elif reason != "exact key":
        status = "equivalent"
    return MappingRow(
        left=left,
        right=right,
        status=status,
        reason=detail,
        layer=_row_layer(left, right),
        modules=_row_modules(left, right),
    )


def _compare_fused_row(
    left: List[TensorInfo],
    right: List[TensorInfo],
    reason: str,
    ignore_quantization: bool = False,
) -> MappingRow:
    quantization_ignored = ignore_quantization and _is_quantization_related(left, right)
    status = "equivalent" if quantization_ignored or _shape_compatible(left, right) else "different"
    detail = reason if status == "equivalent" else f"{reason}; shape mismatch"
    if status == "equivalent" and quantization_ignored:
        detail = f"{reason}; quantization ignored"
    elif status == "equivalent" and not _dtype_equal(left, right):
        detail = f"{reason}; dtype differs"
    return MappingRow(
        left=left,
        right=right,
        status=status,
        reason=detail,
        layer=_row_layer(left, right),
        modules=_row_modules(left, right),
    )


def _shape_equal(left: Sequence[TensorInfo], right: Sequence[TensorInfo]) -> bool:
    return len(left) == len(right) and all(a.shape == b.shape for a, b in zip(left, right))


def _dtype_equal(left: Sequence[TensorInfo], right: Sequence[TensorInfo]) -> bool:
    left_dtypes = {tensor.dtype for tensor in left}
    right_dtypes = {tensor.dtype for tensor in right}
    return left_dtypes == right_dtypes


def _is_quantization_related(left: Sequence[TensorInfo], right: Sequence[TensorInfo]) -> bool:
    tensors = list(left) + list(right)
    return any(
        tensor.kind == "quant_aux"
        or tensor.dtype in {"int4", "uint4", "int8", "i8", "uint8", "u8", "fp8", "f8_e4m3", "f8_e5m2"}
        or tensor.name.endswith((".qweight", ".packed_weight"))
        for tensor in tensors
    )


def _shape_compatible(left: Sequence[TensorInfo], right: Sequence[TensorInfo]) -> bool:
    if len(right) != 1:
        return False
    right_shape = right[0].shape
    if not right_shape:
        return True
    left_shapes = [tensor.shape for tensor in left]
    if any(not shape for shape in left_shapes):
        return True
    if len({len(shape) for shape in left_shapes + [right_shape]}) != 1:
        return False
    if len(right_shape) == 1:
        return sum(shape[0] for shape in left_shapes) == right_shape[0]
    first_dim_sum = sum(shape[0] for shape in left_shapes)
    rest_same = all(shape[1:] == right_shape[1:] for shape in left_shapes)
    if first_dim_sum == right_shape[0] and rest_same:
        return True
    last_dim_sum = sum(shape[-1] for shape in left_shapes)
    prefix_same = all(shape[:-1] == right_shape[:-1] for shape in left_shapes)
    return last_dim_sum == right_shape[-1] and prefix_same


def _by_layer_module(
    tensors: Dict[str, TensorInfo],
    names: Iterable[str],
) -> Dict[int, Dict[str, List[str]]]:
    grouped: Dict[int, Dict[str, List[str]]] = {}
    for name in names:
        tensor = tensors[name]
        if tensor.layer is None:
            continue
        grouped.setdefault(tensor.layer, {}).setdefault(tensor.module, []).append(name)
    return grouped


def _first_name(names: Optional[List[str]]) -> Optional[str]:
    if not names:
        return None
    return sorted(names)[0]


def _row_layer(left: Sequence[TensorInfo], right: Sequence[TensorInfo]) -> Optional[int]:
    for tensor in list(left) + list(right):
        if tensor.layer is not None:
            return tensor.layer
    return None


def _row_modules(left: Sequence[TensorInfo], right: Sequence[TensorInfo]) -> List[str]:
    modules = []
    for tensor in list(left) + list(right):
        if tensor.module and tensor.module not in modules:
            modules.append(tensor.module)
    return modules


def _row_sort_key(row: MappingRow) -> Tuple[int, int, str]:
    layer = -1 if row.layer is None else row.layer
    order = min((_module_order(module) for module in row.modules), default=99)
    name = (row.left_names or row.right_names or [""])[0]
    return (layer, order, name)


def _module_order(module: str) -> int:
    return {
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
    }.get(module, 50)
