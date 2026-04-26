from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from metageniuses.sae.analyze import (
    ActivationBundle,
    _bh_fdr,
    _is_informative_kmer,
    align_records,
    compute_differential_signature,
    compute_feature_stats,
    load_dataset,
    run_probe,
)


def test_load_dataset_supports_nested_label_field(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sequence_id": "seq1",
                        "sequence": "ACTG",
                        "metadata": {"class": "class-1", "source": "foo"},
                    }
                ),
                json.dumps(
                    {
                        "sequence_id": "seq2",
                        "sequence": "TGCA",
                        "metadata": {"class": "class-2", "source": "bar"},
                    }
                ),
            ]
        )
        + "\n"
    )

    df = load_dataset(dataset_path, label_field="metadata.class", positive_label="class-1")
    assert df["sequence_id"].tolist() == ["seq1", "seq2"]
    assert df["y"].tolist() == [1, 0]


def test_align_records_raises_for_missing_ids() -> None:
    dataset_df = pd.DataFrame(
        [
            {"sequence_id": "seq1", "sequence": "AAAA", "label_raw": "class-1", "y": 1, "metadata_json": "{}"}
        ]
    )
    bundle = ActivationBundle(
        matrix=sp.csr_matrix(np.array([[1.0, 0.0]], dtype=np.float32)),
        sequence_ids=["seq2"],
        source_path=Path("/tmp/features.npy"),
        description="test",
    )

    try:
        align_records(dataset_df, bundle)
    except ValueError as exc:
        assert "not found in the dataset" in str(exc)
    else:
        raise AssertionError("Expected align_records to raise on missing sequence IDs")


def test_compute_feature_stats_and_probe_skip_single_class(tmp_path: Path) -> None:
    X = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32))
    y = np.array([1, 1], dtype=np.int8)

    stats = compute_feature_stats(X, y)
    assert stats["test_used"].eq("unavailable_single_class").all()
    assert stats["mean_activation_all"].tolist() == [0.5, 1.0]

    probe = run_probe(X, y, tmp_path, top_n_features=2)
    assert probe.status == "skipped"
    assert (tmp_path / "probe_metrics.csv").exists()


def test_bh_fdr_is_monotonic() -> None:
    adjusted = _bh_fdr(np.array([0.01, 0.04, 0.03, 0.002]))
    assert np.all(adjusted >= 0)
    assert np.all(adjusted <= 1)


def test_kmer_filter_excludes_short_and_low_complexity_patterns() -> None:
    assert not _is_informative_kmer("ATG", min_length=5, min_unique_bases=3, min_entropy=1.75)
    assert not _is_informative_kmer("ATATA", min_length=5, min_unique_bases=3, min_entropy=1.75)
    assert not _is_informative_kmer("AAAAA", min_length=5, min_unique_bases=3, min_entropy=1.75)
    assert _is_informative_kmer("ACGTA", min_length=5, min_unique_bases=3, min_entropy=1.75)


def test_differential_signature_zeroes_nonsignificant_components() -> None:
    feature_stats = pd.DataFrame(
        {
            "feature_idx": [0, 1, 2],
            "mean_activation_pos": [0.5, 0.2, 0.1],
            "mean_activation_neg": [0.1, 0.1, 0.2],
            "fdr_bh": [0.01, 0.20, 0.049],
        }
    )

    signature = compute_differential_signature(feature_stats, fdr_alpha=0.05)
    assert signature.n_significant == 2
    assert np.allclose(signature.difference, np.array([0.4, 0.1, -0.1], dtype=np.float32))
    assert np.allclose(signature.significant_difference, np.array([0.4, 0.0, -0.1], dtype=np.float32))
