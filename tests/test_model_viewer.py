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


if __name__ == "__main__":
    unittest.main()
