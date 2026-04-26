import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from metageniuses.extraction.config import ExtractionConfig
from metageniuses.extraction.extractor import ResidualExtractionPipeline
from metageniuses.extraction.model_adapter import FakeModelAdapter


class TestPipelineFake(unittest.TestCase):
    def test_runs_with_selected_layers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "reads.jsonl"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps({"sequence_id": "r1", "sequence": "ACGTAC"}),
                        json.dumps({"sequence_id": "r2", "sequence": "GGGTTT"}),
                        json.dumps({"sequence_id": "r3", "sequence": "acgtnnxx"}),
                    ]
                )
                + "\n"
            )
            cfg = ExtractionConfig.from_dict(
                {
                    "input": {
                        "path": str(input_path),
                        "format": "jsonl",
                        "sequence_key": "sequence",
                        "id_key": "sequence_id",
                    },
                    "preprocess": {
                        "uppercase": True,
                        "allowed_chars": "ACGTUN",
                        "replace_invalid_with": "N",
                        "max_invalid_fraction": 0.30,
                        "min_length": 1,
                        "max_length": 128,
                        "strip_whitespace": True,
                    },
                    "model": {
                        "model_id": "fake/metagene-tiny",
                        "local_files_only": True,
                    },
                    "layer_selection": {
                        "last_n_layers": 2,
                    },
                    "runtime": {
                        "output_root": str(Path(tmpdir) / "out"),
                        "run_id": "test_run",
                        "batch_size": 2,
                        "max_rows_per_shard": 10000,
                    },
                }
            )

            pipeline = ResidualExtractionPipeline()
            adapter = FakeModelAdapter(num_transformer_layers=8, d_model=12)
            artifact_root = pipeline.run(cfg, adapter=adapter)

            manifest = json.loads((artifact_root / "manifest.json").read_text())
            self.assertEqual(manifest["layer_selection"]["selected_transformer_layers"], [7, 8])
            self.assertEqual(manifest["model"]["d_model"], 12)
            self.assertEqual(manifest["stats"]["total_sequences_kept"], 3)
            self.assertGreater(manifest["stats"]["total_rows_written"], 0)

            layer7_rows = manifest["layers"]["7"]["rows"]
            layer8_rows = manifest["layers"]["8"]["rows"]
            self.assertEqual(layer7_rows, layer8_rows)
            self.assertEqual(layer7_rows * 2, manifest["stats"]["total_rows_written"])

            index_state = json.loads((artifact_root / "_index_state.json").read_text())
            self.assertEqual(index_state["status"], "complete")

            for layer_key, layer_payload in manifest["layers"].items():
                for shard in layer_payload["shards"]:
                    self.assertTrue((artifact_root / "activations" / shard["index_file"]).exists())

            with (artifact_root / "sequences.jsonl").open("r", encoding="utf-8") as handle:
                first_row = json.loads(handle.readline())
            self.assertIn("token_ids", first_row)

    def test_can_disable_deferred_indexing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "reads.jsonl"
            input_path.write_text(json.dumps({"sequence_id": "r1", "sequence": "ACGTAC"}) + "\n")
            cfg = ExtractionConfig.from_dict(
                {
                    "input": {"path": str(input_path), "format": "jsonl"},
                    "preprocess": {"max_invalid_fraction": 1.0, "max_length": 64},
                    "model": {"model_id": "fake/model"},
                    "layer_selection": {"layers": [2]},
                    "runtime": {
                        "output_root": str(Path(tmpdir) / "out"),
                        "run_id": "no_defer",
                        "batch_size": 1,
                        "defer_token_index": False,
                    },
                }
            )

            artifact_root = ResidualExtractionPipeline().run(
                cfg, adapter=FakeModelAdapter(num_transformer_layers=4, d_model=8)
            )
            self.assertFalse((artifact_root / "_index_state.json").exists())
            with (artifact_root / "sequences.jsonl").open("r", encoding="utf-8") as handle:
                first_row = json.loads(handle.readline())
            self.assertNotIn("token_ids", first_row)


if __name__ == "__main__":
    unittest.main()
