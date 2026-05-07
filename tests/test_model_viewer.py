import json
from pathlib import Path
import tempfile
import unittest

from model_viewer.diff import compare_models
from model_viewer.key_patterns import fold_key_patterns
from model_viewer.parsing import load_model
from model_viewer.rendering import render_diff, render_show
from model_viewer.schema import ModelSnapshot, TensorInfo


FIXTURES = Path(__file__).parent / "fixtures"


def qwen35_snapshot() -> ModelSnapshot:
    return ModelSnapshot(
        name="qwen3_5_hybrid",
        source="fixture",
        profile={
            "model_type": "qwen3_5",
            "total_params": 1000,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "vocab_size": 128,
            "tie_word_embeddings": True,
            "layer_kinds": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "num_linear_attn_layers": 3,
            "num_standard_attn_layers": 1,
            "num_kv_cache_layers": 1,
            "full_attention_interval": 4,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_attention_params_per_layer": 128,
            "standard_attention_params_per_layer": 64,
            "attn_output_gate": True,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 4,
            "shared_expert_intermediate_size": 4,
            "is_multimodal": True,
            "vit": {
                "depth": 2,
                "hidden_size": 12,
                "num_heads": 3,
                "intermediate_size": 24,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "out_hidden_size": 16,
                "params": 100,
            },
            "mtp_num_layers": 1,
        },
    )


class ModelViewerTest(unittest.TestCase):
    def test_config_only_model_is_synthesized(self):
        snapshot = load_model(str(FIXTURES / "model_a"))

        self.assertEqual(snapshot.profile["hidden_size"], 16)
        self.assertEqual(snapshot.profile["num_hidden_layers"], 2)
        self.assertTrue(any(tensor.name == "model.layers.0.self_attn.q_proj.weight" for tensor in snapshot.tensors))
        self.assertTrue(snapshot.warnings)

    def test_fuzzy_diff_detects_fused_and_tied_outputs(self):
        left = load_model(str(FIXTURES / "model_a"))
        right = load_model(str(FIXTURES / "fused_b.json"))

        diff = compare_models(left, right, fuzzy_match=True)
        reasons = [row.reason for row in diff.rows]

        self.assertTrue(any("fused q+k+v" in reason for reason in reasons))
        self.assertTrue(any("fused gate+up" in reason for reason in reasons))
        self.assertTrue(any("tied with embedding" in reason for reason in reasons))
        self.assertEqual(diff.summary()["auxiliary"], 1)

    def test_diff_can_ignore_quantization_noise(self):
        left = ModelSnapshot(
            name="bf16",
            source="fixture",
            tensors=[
                TensorInfo(
                    name="model.layers.0.self_attn.o_proj.weight",
                    shape=(16, 16),
                    dtype="bf16",
                    layer=0,
                    module="o_proj",
                    canonical_name="model.layers.0.self_attn.o_proj.weight",
                ),
                TensorInfo(
                    name="model.layers.0.mlp.down_proj.weight",
                    shape=(16, 32),
                    dtype="bf16",
                    layer=0,
                    module="down",
                    canonical_name="model.layers.0.mlp.down_proj.weight",
                ),
            ],
        )
        right = ModelSnapshot(
            name="int4",
            source="fixture",
            tensors=[
                TensorInfo(
                    name="model.layers.0.self_attn.o_proj.qweight",
                    shape=(16, 2),
                    dtype="int4",
                    layer=0,
                    module="o_proj",
                    canonical_name="model.layers.0.self_attn.o_proj.weight",
                ),
                TensorInfo(
                    name="model.layers.0.mlp.down_proj.weight",
                    shape=(16, 32),
                    dtype="int4",
                    layer=0,
                    module="down",
                    canonical_name="model.layers.0.mlp.down_proj.weight",
                ),
                TensorInfo(
                    name="model.layers.0.self_attn.o_proj.scales",
                    shape=(16,),
                    dtype="fp16",
                    kind="quant_aux",
                    layer=0,
                    module="o_proj",
                    canonical_name="model.layers.0.self_attn.o_proj.weight",
                    parent="model.layers.0.self_attn.o_proj.weight",
                ),
            ],
        )

        diff = compare_models(left, right, ignore_quantization=True)

        self.assertFalse(diff.has_change)
        self.assertEqual(diff.summary()["auxiliary"], 0)
        self.assertEqual(diff.summary()["different"], 0)
        self.assertTrue(all(row.status == "exact" for row in diff.rows))
        self.assertTrue(all("quantization ignored" in row.reason for row in diff.rows))

    def test_renderers_produce_core_views(self):
        left = load_model(str(FIXTURES / "model_a"))
        right = load_model(str(FIXTURES / "fused_b.json"))
        diff = compare_models(left, right, fuzzy_match=True)

        self.assertIn("flowchart TB", render_show(left, ["overview"], "mermaid"))
        report = render_diff(diff, ["heatmap", "mapping", "memory"], "markdown")
        self.assertIn("Heatmap", report)
        self.assertIn("Key Mapping", report)
        self.assertIn("Memory Footprint", report)

    def test_key_patterns_fold_numeric_positions(self):
        snapshot = ModelSnapshot(
            name="moe",
            source="fixture",
            tensors=[
                TensorInfo(name=f"model.layers.0.mlp.experts.{idx}.weight", shape=(8, 4), dtype="bf16")
                for idx in range(4)
            ],
        )

        patterns = fold_key_patterns(snapshot)
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0].pattern, "model.layers.{0}.mlp.experts.{0..3}.weight")
        self.assertEqual(patterns[0].count, 4)
        rendered = render_show(snapshot, ["patterns"], "markdown")
        self.assertIn("Safetensor Key Patterns", rendered)
        self.assertIn("model.layers.{0}.mlp.experts.{0..3}.weight", rendered)

    def test_diff_patterns_highlight_fused_pattern_changes(self):
        left_tensors = []
        right_tensors = []
        for layer in (0, 1):
            prefix = f"model.language_model.layers.{layer}"
            left_tensors.extend(
                [
                    TensorInfo(name=f"{prefix}.linear_attn.in_proj_qkv.weight", shape=(8192, 2048), dtype="bf16"),
                    TensorInfo(name=f"{prefix}.linear_attn.in_proj_z.weight", shape=(4096, 2048), dtype="bf16"),
                    TensorInfo(name=f"{prefix}.linear_attn.in_proj_b.weight", shape=(32, 2048), dtype="bf16"),
                    TensorInfo(name=f"{prefix}.linear_attn.in_proj_a.weight", shape=(32, 2048), dtype="bf16"),
                    TensorInfo(name=f"{prefix}.linear_attn.A_log", shape=(32,), dtype="bf16"),
                    TensorInfo(name=f"{prefix}.linear_attn.norm.weight", shape=(128,), dtype="bf16"),
                    TensorInfo(name=f"{prefix}.mlp.experts.gate_up_proj", shape=(2, 1024, 2048), dtype="bf16"),
                    TensorInfo(name=f"{prefix}.mlp.experts.down_proj", shape=(2, 2048, 512), dtype="bf16"),
                ]
            )
            right_tensors.extend(
                [
                    TensorInfo(name=f"{prefix}.linear_attn.in_proj_qkvz.weight", shape=(12288, 2048), dtype="bf16"),
                    TensorInfo(name=f"{prefix}.linear_attn.in_proj_ba.weight", shape=(64, 2048), dtype="bf16"),
                    TensorInfo(name=f"{prefix}.linear_attn.A_log", shape=(32,), dtype="f32"),
                    TensorInfo(name=f"{prefix}.linear_attn.norm.weight", shape=(128,), dtype="f32"),
                ]
            )
            for expert in range(2):
                right_tensors.extend(
                    [
                        TensorInfo(name=f"{prefix}.mlp.experts.{expert}.gate_proj.weight", shape=(512, 2048), dtype="bf16"),
                        TensorInfo(name=f"{prefix}.mlp.experts.{expert}.up_proj.weight", shape=(512, 2048), dtype="bf16"),
                        TensorInfo(name=f"{prefix}.mlp.experts.{expert}.down_proj.weight", shape=(2048, 512), dtype="bf16"),
                    ]
                )

        diff = compare_models(
            ModelSnapshot(name="left", source="fixture", tensors=left_tensors),
            ModelSnapshot(name="right", source="fixture", tensors=right_tensors),
        )
        rendered = render_diff(diff, ["patterns"], "markdown")

        self.assertIn("Safetensor Key Pattern Diff", rendered)
        self.assertIn("Interpretation:", rendered)
        self.assertIn("different storage layout, same logical tensors", rendered)
        self.assertIn("same key pattern and shape/count, but stored dtype differs", rendered)
        self.assertIn("Conclusion: no unmatched pattern remains", rendered)
        self.assertIn("linear_attn fusion: in_proj_qkv + in_proj_z -> in_proj_qkvz", rendered)
        self.assertIn("linear_attn fusion: in_proj_b + in_proj_a -> in_proj_ba", rendered)
        self.assertIn("MoE expert fusion: expert gate_proj + up_proj -> gate_up_proj", rendered)
        self.assertIn("MoE expert packing: experts dimension packed into one tensor", rendered)
        self.assertIn("dtype differs: bf16 -> f32", rendered)
        self.assertNotIn("Safetensor Key Patterns: left", rendered)

    def test_block_diagram_renders_structure_blocks(self):
        snapshot = load_model(str(FIXTURES / "model_a"))

        rendered = render_show(snapshot, ["blocks"], "markdown")
        self.assertIn("Character Block Diagram", rendered)
        self.assertIn("TOKEN EMBEDDING", rendered)
        self.assertIn("LANGUAGE DECODER STACK x 2", rendered)
        self.assertIn("Attention", rendered)
        self.assertIn("MLP", rendered)

    def test_block_diagram_renders_qwen35_hybrid_structure(self):
        snapshot = qwen35_snapshot()

        rendered = render_show(snapshot, ["blocks"], "markdown")
        self.assertIn("MULTIMODAL INPUT ROUTER", rendered)
        self.assertIn("HYBRID LAYER SCHEDULE", rendered)
        self.assertIn("macro-block x1: [L1:DeltaNet -> L2:DeltaNet -> L3:DeltaNet -> L4:GQA]", rendered)
        self.assertIn("DeltaNet layers: {0..2}", rendered)
        self.assertIn("GQA layers: {3}", rendered)
        self.assertIn("KV Cache layers=1; State Cache layers=3", rendered)
        self.assertIn("DeltaNet / linear_attn x 3", rendered)
        self.assertIn("GQA / self_attn x 1", rendered)
        self.assertIn("SwiGLU MoE MLP", rendered)
        self.assertIn("experts=8", rendered)
        self.assertIn("active=2", rendered)
        self.assertIn("MTP AUXILIARY HEAD", rendered)

    def test_qwen35_views_render_hybrid_details_beyond_blocks(self):
        snapshot = qwen35_snapshot()

        overview = render_show(snapshot, ["overview"], "markdown")
        tree = render_show(snapshot, ["tree"], "markdown")
        delta_detail = render_show(snapshot, ["detail"], "markdown", layer=0)
        gqa_detail = render_show(snapshot, ["detail"], "markdown", layer=3)
        memory = render_show(snapshot, ["memory"], "markdown")

        self.assertIn("Hybrid Schedule", overview)
        self.assertIn("DeltaNet / linear_attn x 3", overview)
        self.assertIn("Vision Encoder", overview)
        self.assertIn("DeltaNet layers {0..2}", tree)
        self.assertIn("GQA layers {3}", tree)
        self.assertIn("State Cache", tree)
        self.assertIn("LANGUAGE LAYER 0 [DeltaNet / linear_attention]", delta_detail)
        self.assertIn("linear_attn in_proj_qkv", delta_detail)
        self.assertIn("LANGUAGE LAYER 3 [GQA / full_attention]", gqa_detail)
        self.assertIn("GQA self_attn", gqa_detail)
        self.assertIn("Vision Encoder", memory)
        self.assertIn("State Cache", memory)

    def test_qwen35_moe_reads_layer_types_from_config_and_index(self):
        layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 10
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5_moe",
                        "text_config": {
                            "model_type": "qwen3_5_moe_text",
                            "hidden_size": 2048,
                            "num_hidden_layers": 40,
                            "num_attention_heads": 16,
                            "num_key_value_heads": 2,
                            "head_dim": 256,
                            "vocab_size": 248320,
                            "layer_types": layer_types,
                            "full_attention_interval": 4,
                            "linear_num_key_heads": 16,
                            "linear_num_value_heads": 32,
                            "linear_key_head_dim": 128,
                            "linear_value_head_dim": 128,
                            "attn_output_gate": True,
                            "num_experts": 256,
                            "num_experts_per_tok": 8,
                            "moe_intermediate_size": 512,
                            "shared_expert_intermediate_size": 512,
                        },
                    }
                ),
                encoding="utf-8",
            )
            weight_map = {
                "model.language_model.embed_tokens.weight": "model-00001-of-00001.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": "model-00001-of-00001.safetensors",
                "model.language_model.layers.0.linear_attn.out_proj.weight": "model-00001-of-00001.safetensors",
                "model.language_model.layers.3.self_attn.q_proj.weight": "model-00001-of-00001.safetensors",
                "model.language_model.layers.3.self_attn.k_proj.weight": "model-00001-of-00001.safetensors",
                "model.language_model.layers.3.self_attn.v_proj.weight": "model-00001-of-00001.safetensors",
                "model.language_model.layers.3.self_attn.o_proj.weight": "model-00001-of-00001.safetensors",
                "model.language_model.layers.0.mlp.gate.weight": "model-00001-of-00001.safetensors",
                "model.language_model.layers.0.mlp.shared_expert.down_proj.weight": "model-00001-of-00001.safetensors",
            }
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")

            snapshot = load_model(str(root))
            rendered = render_show(snapshot, ["blocks", "patterns"], "markdown")

        self.assertEqual(snapshot.profile["num_linear_attn_layers"], 30)
        self.assertEqual(snapshot.profile["num_standard_attn_layers"], 10)
        self.assertIn("HYBRID LAYER SCHEDULE", rendered)
        self.assertIn("DeltaNet/linear=30  GQA/full=10", rendered)
        self.assertIn("in_proj_qkv  [8192,2048]", rendered)
        self.assertIn("experts=256", rendered)
        self.assertIn("shared_expert intermediate=512", rendered)
        self.assertIn("Architecture: DeltaNet layers", rendered)

    def test_qwen35_moe_without_layer_metadata_is_not_guessed(self):
        snapshot = ModelSnapshot(
            name="missing_metadata",
            source="fixture",
            profile={
                "model_type": "qwen3_5_moe",
                "hidden_size": 2048,
                "num_hidden_layers": 40,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "vocab_size": 248320,
                "num_experts": 256,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 512,
                "layer_kinds_source": "unspecified",
            },
        )

        rendered = render_show(snapshot, ["blocks"], "markdown")

        self.assertIn("LAYER TYPE METADATA MISSING", rendered)
        self.assertNotIn("HYBRID LAYER SCHEDULE", rendered)


if __name__ == "__main__":
    unittest.main()
