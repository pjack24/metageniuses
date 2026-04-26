import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from metageniuses.extraction.config import ExtractionConfig
from metageniuses.extraction.extractor import ResidualExtractionPipeline
from metageniuses.extraction.model_adapter import FakeModelAdapter


class RecordingBatchSizeAdapter(FakeModelAdapter):
    def __init__(self) -> None:
        super().__init__(num_transformer_layers=8, d_model=8)
        self.batch_sizes: list[int] = []

    def extract_batch(
        self,
        sequences: list[str],
        transformer_layers: list[int],
        max_length: int,
    ):
        self.batch_sizes.append(len(sequences))
        return super().extract_batch(sequences, transformer_layers, max_length)


class OOMOnLargeBatchAdapter(FakeModelAdapter):
    def __init__(self, max_supported_batch: int) -> None:
        super().__init__(num_transformer_layers=8, d_model=8)
        self.max_supported_batch = max_supported_batch
        self.batch_sizes: list[int] = []
        self.attempted_batch_sizes: list[int] = []

    def extract_batch(
        self,
        sequences: list[str],
        transformer_layers: list[int],
        max_length: int,
    ):
        self.attempted_batch_sizes.append(len(sequences))
        if len(sequences) > self.max_supported_batch:
            raise RuntimeError("CUDA out of memory")
        self.batch_sizes.append(len(sequences))
        return super().extract_batch(sequences, transformer_layers, max_length)


class TestAdaptiveBatching(unittest.TestCase):
    def _write_reads(self, path: Path, count: int) -> None:
        lines = []
        for idx in range(count):
            lines.append(json.dumps({"sequence_id": f"r{idx}", "sequence": "ACGTACGT"}))
        path.write_text("\n".join(lines) + "\n")

    def test_batch_size_grows_to_max(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "reads.jsonl"
            self._write_reads(input_path, 10)
            cfg = ExtractionConfig.from_dict(
                {
                    "input": {"path": str(input_path), "format": "jsonl"},
                    "preprocess": {"max_invalid_fraction": 1.0},
                    "model": {"model_id": "fake/model"},
                    "layer_selection": {"layers": [2]},
                    "runtime": {
                        "output_root": str(Path(tmpdir) / "out"),
                        "run_id": "adaptive_growth",
                        "batch_size": 2,
                        "max_batch_size": 4,
                        "batch_growth_success_batches": 1,
                        "max_rows_per_shard": 10000,
                    },
                }
            )
            adapter = RecordingBatchSizeAdapter()
            artifact_root = ResidualExtractionPipeline().run(cfg, adapter=adapter)
            manifest = json.loads((artifact_root / "manifest.json").read_text())

            self.assertEqual(manifest["stats"]["total_sequences_kept"], 10)
            self.assertIn(4, adapter.batch_sizes)

    def test_oom_reduces_batch_and_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "reads.jsonl"
            self._write_reads(input_path, 6)
            cfg = ExtractionConfig.from_dict(
                {
                    "input": {"path": str(input_path), "format": "jsonl"},
                    "preprocess": {"max_invalid_fraction": 1.0},
                    "model": {"model_id": "fake/model"},
                    "layer_selection": {"layers": [2]},
                    "runtime": {
                        "output_root": str(Path(tmpdir) / "out"),
                        "run_id": "adaptive_oom",
                        "batch_size": 4,
                        "max_batch_size": 4,
                        "batch_growth_success_batches": 1,
                        "reduce_batch_on_oom": True,
                        "max_rows_per_shard": 10000,
                    },
                }
            )
            adapter = OOMOnLargeBatchAdapter(max_supported_batch=2)
            artifact_root = ResidualExtractionPipeline().run(cfg, adapter=adapter)
            manifest = json.loads((artifact_root / "manifest.json").read_text())

            self.assertEqual(manifest["stats"]["total_sequences_kept"], 6)
            self.assertTrue(adapter.batch_sizes)
            self.assertTrue(all(size <= 2 for size in adapter.batch_sizes))

    def test_batch_growth_step_is_gradual(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "reads.jsonl"
            self._write_reads(input_path, 15)
            cfg = ExtractionConfig.from_dict(
                {
                    "input": {"path": str(input_path), "format": "jsonl"},
                    "preprocess": {"max_invalid_fraction": 1.0},
                    "model": {"model_id": "fake/model"},
                    "layer_selection": {"layers": [2]},
                    "runtime": {
                        "output_root": str(Path(tmpdir) / "out"),
                        "run_id": "adaptive_step_growth",
                        "batch_size": 2,
                        "max_batch_size": 6,
                        "batch_growth_success_batches": 1,
                        "batch_growth_step": 2,
                        "max_rows_per_shard": 10000,
                    },
                }
            )
            adapter = RecordingBatchSizeAdapter()
            artifact_root = ResidualExtractionPipeline().run(cfg, adapter=adapter)
            manifest = json.loads((artifact_root / "manifest.json").read_text())

            self.assertEqual(manifest["stats"]["total_sequences_kept"], 15)
            self.assertIn(4, adapter.batch_sizes)
            self.assertIn(6, adapter.batch_sizes)

    def test_initial_max_batch_cap_releases_after_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "reads.jsonl"
            self._write_reads(input_path, 20)
            cfg = ExtractionConfig.from_dict(
                {
                    "input": {"path": str(input_path), "format": "jsonl"},
                    "preprocess": {"max_invalid_fraction": 1.0},
                    "model": {"model_id": "fake/model"},
                    "layer_selection": {"layers": [2]},
                    "runtime": {
                        "output_root": str(Path(tmpdir) / "out"),
                        "run_id": "adaptive_initial_cap",
                        "batch_size": 2,
                        "max_batch_size": 6,
                        "initial_max_batch_size": 4,
                        "release_to_max_after_sequences": 8,
                        "batch_growth_success_batches": 1,
                        "batch_growth_step": 2,
                        "max_rows_per_shard": 10000,
                    },
                }
            )
            adapter = RecordingBatchSizeAdapter()
            artifact_root = ResidualExtractionPipeline().run(cfg, adapter=adapter)
            manifest = json.loads((artifact_root / "manifest.json").read_text())

            self.assertEqual(manifest["stats"]["total_sequences_kept"], 20)
            self.assertIn(6, adapter.batch_sizes)
            first_six = adapter.batch_sizes.index(6)
            self.assertTrue(all(size <= 4 for size in adapter.batch_sizes[:first_six]))

    def test_initial_max_batch_cap_does_not_release_after_oom(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "reads.jsonl"
            self._write_reads(input_path, 20)
            cfg = ExtractionConfig.from_dict(
                {
                    "input": {"path": str(input_path), "format": "jsonl"},
                    "preprocess": {"max_invalid_fraction": 1.0},
                    "model": {"model_id": "fake/model"},
                    "layer_selection": {"layers": [2]},
                    "runtime": {
                        "output_root": str(Path(tmpdir) / "out"),
                        "run_id": "adaptive_initial_cap_no_release_after_oom",
                        "batch_size": 2,
                        "max_batch_size": 6,
                        "initial_max_batch_size": 4,
                        "release_to_max_after_sequences": 8,
                        "batch_growth_success_batches": 1,
                        "batch_growth_step": 2,
                        "reduce_batch_on_oom": True,
                        "max_rows_per_shard": 10000,
                    },
                }
            )
            adapter = OOMOnLargeBatchAdapter(max_supported_batch=3)
            artifact_root = ResidualExtractionPipeline().run(cfg, adapter=adapter)
            manifest = json.loads((artifact_root / "manifest.json").read_text())

            self.assertEqual(manifest["stats"]["total_sequences_kept"], 20)
            self.assertIn(4, adapter.attempted_batch_sizes)
            self.assertNotIn(6, adapter.attempted_batch_sizes)
            self.assertTrue(all(size <= 3 for size in adapter.batch_sizes))


if __name__ == "__main__":
    unittest.main()
