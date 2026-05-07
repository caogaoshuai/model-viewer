from pathlib import Path
import unittest

from model_viewer.diff import compare_models
from model_viewer.key_patterns import fold_key_patterns
from model_viewer.parsing import load_model
from model_viewer.rendering import render_diff, render_show
from model_viewer.schema import ModelSnapshot, TensorInfo


FIXTURES = Path(__file__).parent / "fixtures"


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

    def test_block_diagram_renders_structure_blocks(self):
        snapshot = load_model(str(FIXTURES / "model_a"))

        rendered = render_show(snapshot, ["blocks"], "markdown")
        self.assertIn("Character Block Diagram", rendered)
        self.assertIn("TOKEN EMBEDDING", rendered)
        self.assertIn("LANGUAGE DECODER STACK x 2", rendered)
        self.assertIn("Attention", rendered)
        self.assertIn("MLP", rendered)

    def test_block_diagram_renders_qwen35_hybrid_structure(self):
        snapshot = ModelSnapshot(
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
                },
                "mtp_num_layers": 1,
            },
        )

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


if __name__ == "__main__":
    unittest.main()
