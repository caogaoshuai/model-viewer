from pathlib import Path
import unittest

from model_viewer.diff import compare_models
from model_viewer.parsing import load_model
from model_viewer.rendering import render_diff, render_show


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


if __name__ == "__main__":
    unittest.main()
