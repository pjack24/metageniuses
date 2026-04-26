from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler


SUPPORTED_ACTIVATION_SUFFIXES = {".npy", ".npz", ".pt", ".jsonl"}
SEQUENCE_ID_CANDIDATES = (
    "sequence_ids.json",
    "sequence_ids.jsonl",
    "sequence_ids.txt",
)


@dataclass
class ActivationBundle:
    matrix: sp.csr_matrix
    sequence_ids: list[str]
    source_path: Path
    description: str


@dataclass
class ProbeResult:
    metrics: pd.DataFrame
    coefficients: pd.DataFrame
    top_positive: pd.DataFrame
    top_negative: pd.DataFrame
    status: str
    message: str


@dataclass
class DifferentialSignature:
    table: pd.DataFrame
    positive_mean: np.ndarray
    negative_mean: np.ndarray
    difference: np.ndarray
    significant_difference: np.ndarray
    n_significant: int
    fdr_alpha: float


def _get_nested(record: dict[str, Any], field: str) -> Any:
    value: Any = record
    for part in field.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
            continue
        raise KeyError(field)
    return value


def _infer_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(record.get("metadata", {})) if isinstance(record.get("metadata"), dict) else {}
    for key, value in record.items():
        if key in {"sequence_id", "sequence", "metadata"}:
            continue
        metadata.setdefault(key, value)
    return metadata


def load_dataset(
    dataset_path: str | Path,
    label_field: str,
    positive_label: str,
) -> pd.DataFrame:
    dataset_path = Path(dataset_path)
    rows: list[dict[str, Any]] = []
    missing_labels = 0

    with dataset_path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            sequence_id = record.get("sequence_id")
            sequence = record.get("sequence")
            if not sequence_id:
                raise ValueError(f"{dataset_path}:{line_no} is missing 'sequence_id'")
            if sequence is None:
                raise ValueError(f"{dataset_path}:{line_no} is missing 'sequence'")

            try:
                label_value = _get_nested(record, label_field)
            except KeyError:
                missing_labels += 1
                continue

            rows.append(
                {
                    "sequence_id": str(sequence_id),
                    "sequence": str(sequence),
                    "label_raw": str(label_value),
                    "y": int(str(label_value) == positive_label),
                    "metadata_json": json.dumps(_infer_metadata(record), sort_keys=True),
                }
            )

    if not rows:
        raise ValueError(
            f"No labeled rows were loaded from {dataset_path}. "
            f"Checked field '{label_field}' and found no usable labels."
        )

    if missing_labels:
        print(
            f"Warning: skipped {missing_labels} rows from {dataset_path} because "
            f"'{label_field}' was missing."
        )

    return pd.DataFrame(rows)


def _discover_sequence_id_file(base_path: Path) -> Path | None:
    search_dir = base_path if base_path.is_dir() else base_path.parent
    for candidate in SEQUENCE_ID_CANDIDATES:
        path = search_dir / candidate
        if path.exists():
            return path
    return None


def _load_sequence_ids(path: Path) -> list[str]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a JSON list of sequence IDs")
        return [str(item) for item in payload]

    if path.suffix == ".jsonl":
        sequence_ids: list[str] = []
        with path.open() as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                sequence_id = record.get("sequence_id")
                if sequence_id is None:
                    raise ValueError(f"{path}:{line_no} is missing 'sequence_id'")
                sequence_ids.append(str(sequence_id))
        return sequence_ids

    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _resolve_activation_file(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path

    if not path.is_dir():
        raise FileNotFoundError(f"Activation path does not exist: {path}")

    preferred = ("features.npy", "features.npz", "features.pt", "activations.npz", "activations.pt")
    for candidate in preferred:
        candidate_path = path / candidate
        if candidate_path.exists():
            return candidate_path

    supported = sorted(
        candidate for candidate in path.iterdir() if candidate.suffix in SUPPORTED_ACTIVATION_SUFFIXES
    )
    if len(supported) == 1:
        return supported[0]
    if not supported:
        raise FileNotFoundError(
            f"No activation file found in {path}. Expected one of: {sorted(SUPPORTED_ACTIVATION_SUFFIXES)}"
        )
    raise ValueError(
        f"Activation directory {path} contains multiple supported files. "
        f"Pass one file explicitly: {[str(item.name) for item in supported]}"
    )


def _row_topk_to_csr(indices: np.ndarray, values: np.ndarray, n_features: int) -> sp.csr_matrix:
    if indices.shape != values.shape:
        raise ValueError("top-k activation indices and values must have the same shape")

    n_rows, top_k = indices.shape
    indptr = np.arange(0, (n_rows + 1) * top_k, top_k, dtype=np.int64)
    matrix = sp.csr_matrix(
        (values.reshape(-1), indices.reshape(-1), indptr),
        shape=(n_rows, n_features),
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix


def _dense_to_csr(array: np.ndarray, chunk_rows: int = 512) -> sp.csr_matrix:
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D activation array, got shape {array.shape}")

    n_rows, n_features = array.shape
    data_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    indptr = [0]

    for start in range(0, n_rows, chunk_rows):
        stop = min(start + chunk_rows, n_rows)
        chunk = np.asarray(array[start:stop])
        row_idx, col_idx = np.nonzero(chunk)
        values = chunk[row_idx, col_idx].astype(np.float32, copy=False)
        data_parts.append(values)
        index_parts.append(col_idx.astype(np.int32, copy=False))
        counts = np.bincount(row_idx, minlength=chunk.shape[0]).astype(np.int64)
        cumulative = counts.cumsum() + indptr[-1]
        indptr.extend(cumulative.tolist())

    data = np.concatenate(data_parts) if data_parts else np.array([], dtype=np.float32)
    indices = np.concatenate(index_parts) if index_parts else np.array([], dtype=np.int32)
    return sp.csr_matrix((data, indices, np.array(indptr, dtype=np.int64)), shape=(n_rows, n_features))


def _load_numpy_npz(path: Path) -> tuple[sp.csr_matrix, list[str] | None, str]:
    try:
        sparse_matrix = sp.load_npz(path)
        return sparse_matrix.tocsr().astype(np.float32), None, "scipy sparse .npz"
    except Exception:
        pass

    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.lib.npyio.NpzFile):
        keys = set(payload.files)
        if {"data", "indices", "indptr", "shape"} <= keys:
            matrix = sp.csr_matrix(
                (
                    payload["data"].astype(np.float32, copy=False),
                    payload["indices"],
                    payload["indptr"],
                ),
                shape=tuple(payload["shape"]),
            )
            ids = payload["sequence_ids"].tolist() if "sequence_ids" in keys else None
            return matrix, [str(item) for item in ids] if ids is not None else None, "csr parts .npz"

        dense_key = next((key for key in ("features", "activations", "x") if key in keys), None)
        if dense_key is not None:
            dense = payload[dense_key]
            ids = payload["sequence_ids"].tolist() if "sequence_ids" in keys else None
            return _dense_to_csr(dense), [str(item) for item in ids] if ids is not None else None, f"dense key '{dense_key}' .npz"

        if {"topk_indices", "topk_values"} <= keys and "d_sae" in keys:
            indices = payload["topk_indices"]
            values = payload["topk_values"].astype(np.float32, copy=False)
            ids = payload["sequence_ids"].tolist() if "sequence_ids" in keys else None
            return (
                _row_topk_to_csr(indices, values, int(payload["d_sae"])),
                [str(item) for item in ids] if ids is not None else None,
                "top-k .npz",
            )

        raise ValueError(f"Could not interpret keys in {path}: {sorted(keys)}")

    return _dense_to_csr(payload), None, "dense .npy"


def _load_torch_pt(path: Path) -> tuple[sp.csr_matrix, list[str] | None, str]:
    import torch

    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        return _dense_to_csr(payload.numpy()), None, "dense tensor .pt"

    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported .pt payload type: {type(payload)}")

    if {"topk_indices", "topk_values"} <= set(payload):
        indices = payload["topk_indices"]
        values = payload["topk_values"]
        n_features = payload.get("d_sae") or payload.get("n_features") or payload.get("shape", [None, None])[1]
        if n_features is None:
            raise ValueError(f"{path} contains top-k activations but no feature dimension")
        ids = payload.get("sequence_ids")
        return (
            _row_topk_to_csr(np.asarray(indices), np.asarray(values, dtype=np.float32), int(n_features)),
            [str(item) for item in ids] if ids is not None else None,
            "top-k .pt",
        )

    dense_key = next((key for key in ("features", "activations", "x", "matrix") if key in payload), None)
    if dense_key is not None:
        tensor = payload[dense_key]
        array = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
        ids = payload.get("sequence_ids")
        return _dense_to_csr(array), [str(item) for item in ids] if ids is not None else None, f"dense key '{dense_key}' .pt"

    if {"data", "indices", "indptr", "shape"} <= set(payload):
        matrix = sp.csr_matrix(
            (
                np.asarray(payload["data"], dtype=np.float32),
                np.asarray(payload["indices"]),
                np.asarray(payload["indptr"]),
            ),
            shape=tuple(payload["shape"]),
        )
        ids = payload.get("sequence_ids")
        return matrix, [str(item) for item in ids] if ids is not None else None, "csr parts .pt"

    raise ValueError(f"Could not interpret .pt payload keys in {path}: {sorted(payload)}")


def _load_jsonl_activations(path: Path) -> tuple[sp.csr_matrix, list[str], str]:
    sequence_ids: list[str] = []
    row_data: list[float] = []
    row_indices: list[int] = []
    indptr = [0]
    inferred_n_features = 0

    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            sequence_id = record.get("sequence_id")
            if sequence_id is None:
                raise ValueError(f"{path}:{line_no} is missing 'sequence_id'")
            sequence_ids.append(str(sequence_id))

            if "topk_indices" in record and "topk_values" in record:
                indices = np.asarray(record["topk_indices"], dtype=np.int32)
                values = np.asarray(record["topk_values"], dtype=np.float32)
            elif "indices" in record and "values" in record:
                indices = np.asarray(record["indices"], dtype=np.int32)
                values = np.asarray(record["values"], dtype=np.float32)
            elif "features" in record:
                dense = np.asarray(record["features"], dtype=np.float32)
                indices = np.nonzero(dense)[0].astype(np.int32)
                values = dense[indices]
            else:
                raise ValueError(
                    f"{path}:{line_no} must contain one of "
                    f"('topk_indices'+'topk_values'), ('indices'+'values'), or 'features'"
                )

            if indices.shape != values.shape:
                raise ValueError(f"{path}:{line_no} has mismatched activation index/value lengths")
            if indices.size:
                inferred_n_features = max(inferred_n_features, int(indices.max()) + 1)
            row_indices.extend(indices.tolist())
            row_data.extend(values.tolist())
            indptr.append(len(row_indices))

    matrix = sp.csr_matrix(
        (
            np.asarray(row_data, dtype=np.float32),
            np.asarray(row_indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(len(sequence_ids), inferred_n_features),
    )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix, sequence_ids, "jsonl activations"


def load_activations(activation_path: str | Path) -> ActivationBundle:
    activation_file = _resolve_activation_file(activation_path)

    if activation_file.suffix == ".npy":
        matrix = _dense_to_csr(np.load(activation_file, mmap_mode="r"))
        sequence_ids = None
        description = "dense .npy"
    elif activation_file.suffix == ".npz":
        matrix, sequence_ids, description = _load_numpy_npz(activation_file)
    elif activation_file.suffix == ".pt":
        matrix, sequence_ids, description = _load_torch_pt(activation_file)
    elif activation_file.suffix == ".jsonl":
        matrix, sequence_ids, description = _load_jsonl_activations(activation_file)
    else:
        raise ValueError(f"Unsupported activation file suffix: {activation_file.suffix}")

    if sequence_ids is None:
        sequence_id_file = _discover_sequence_id_file(activation_file)
        if sequence_id_file is None:
            raise FileNotFoundError(
                f"Could not find sequence IDs next to {activation_file}. "
                f"Expected one of: {list(SEQUENCE_ID_CANDIDATES)}"
            )
        sequence_ids = _load_sequence_ids(sequence_id_file)

    if matrix.shape[0] != len(sequence_ids):
        raise ValueError(
            f"Activation rows ({matrix.shape[0]}) do not match number of sequence IDs "
            f"({len(sequence_ids)}) for {activation_file}"
        )

    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return ActivationBundle(
        matrix=matrix.tocsr().astype(np.float32),
        sequence_ids=sequence_ids,
        source_path=activation_file,
        description=description,
    )


def align_records(dataset_df: pd.DataFrame, activations: ActivationBundle) -> tuple[pd.DataFrame, sp.csr_matrix]:
    dataset_indexed = dataset_df.drop_duplicates("sequence_id").set_index("sequence_id")
    activation_ids = pd.Index(activations.sequence_ids, name="sequence_id")

    missing_in_dataset = activation_ids.difference(dataset_indexed.index)
    missing_in_activations = dataset_indexed.index.difference(activation_ids)

    if len(missing_in_dataset):
        examples = ", ".join(missing_in_dataset[:5].tolist())
        raise ValueError(
            f"{len(missing_in_dataset)} activation sequence IDs were not found in the dataset. "
            f"Examples: {examples}"
        )
    if len(missing_in_activations):
        print(
            f"Warning: dataset contains {len(missing_in_activations)} labeled sequences without activations. "
            f"Only the intersection will be analyzed."
        )

    aligned = dataset_indexed.loc[activation_ids].reset_index()
    return aligned, activations.matrix


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    n = len(ranked)
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        candidate = ranked[i] * n / rank
        prev = min(prev, candidate)
        adjusted[i] = min(prev, 1.0)
    result = np.empty(n, dtype=float)
    result[order] = adjusted
    return result


def compute_feature_stats(X: sp.csr_matrix, y: np.ndarray) -> pd.DataFrame:
    n_samples, n_features = X.shape
    y = np.asarray(y, dtype=np.int8)
    pos_mask = y == 1
    neg_mask = y == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())

    activation_rate = X.getnnz(axis=0) / max(n_samples, 1)
    mean_activation_all = np.asarray(X.mean(axis=0)).ravel()

    feature_stats = pd.DataFrame(
        {
            "feature_idx": np.arange(n_features, dtype=int),
            "activation_rate": activation_rate,
            "mean_activation_all": mean_activation_all,
        }
    )

    if n_pos == 0 or n_neg == 0:
        feature_stats["mean_activation_pos"] = np.nan
        feature_stats["mean_activation_neg"] = np.nan
        feature_stats["fraction_active_pos"] = np.nan
        feature_stats["fraction_active_neg"] = np.nan
        feature_stats["class_enrichment"] = np.nan
        feature_stats["log_odds_ratio"] = np.nan
        feature_stats["p_value"] = np.nan
        feature_stats["fdr_bh"] = np.nan
        feature_stats["test_used"] = "unavailable_single_class"
        return feature_stats

    X_pos = X[pos_mask]
    X_neg = X[neg_mask]
    active_pos = np.asarray(X_pos.getnnz(axis=0), dtype=np.int64)
    active_neg = np.asarray(X_neg.getnnz(axis=0), dtype=np.int64)
    mean_activation_pos = np.asarray(X_pos.mean(axis=0)).ravel()
    mean_activation_neg = np.asarray(X_neg.mean(axis=0)).ravel()
    fraction_active_pos = active_pos / n_pos
    fraction_active_neg = active_neg / n_neg
    class_enrichment = (fraction_active_pos + 1e-6) / (fraction_active_neg + 1e-6)
    log_odds_ratio = np.log(
        ((active_pos + 0.5) / (n_pos - active_pos + 0.5))
        / ((active_neg + 0.5) / (n_neg - active_neg + 0.5))
    )

    p_values = np.ones(n_features, dtype=float)
    test_used: list[str] = []
    for idx in range(n_features):
        table = np.array(
            [
                [active_pos[idx], n_pos - active_pos[idx]],
                [active_neg[idx], n_neg - active_neg[idx]],
            ],
            dtype=float,
        )
        expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
        if (expected < 5).any():
            _, p_value = fisher_exact(table.astype(int))
            test_used.append("fisher_exact")
        else:
            _, p_value, _, _ = chi2_contingency(table, correction=False)
            test_used.append("chi_square")
        p_values[idx] = p_value

    feature_stats["mean_activation_pos"] = mean_activation_pos
    feature_stats["mean_activation_neg"] = mean_activation_neg
    feature_stats["fraction_active_pos"] = fraction_active_pos
    feature_stats["fraction_active_neg"] = fraction_active_neg
    feature_stats["class_enrichment"] = class_enrichment
    feature_stats["log_odds_ratio"] = log_odds_ratio
    feature_stats["p_value"] = p_values
    feature_stats["fdr_bh"] = _bh_fdr(p_values)
    feature_stats["test_used"] = test_used
    return feature_stats


def compute_differential_signature(
    feature_stats: pd.DataFrame,
    fdr_alpha: float,
) -> DifferentialSignature:
    table = feature_stats.copy()

    positive_mean = table["mean_activation_pos"].to_numpy(dtype=np.float32, copy=True)
    negative_mean = table["mean_activation_neg"].to_numpy(dtype=np.float32, copy=True)
    difference = positive_mean - negative_mean

    significant_mask = np.zeros(len(table), dtype=bool)
    if "fdr_bh" in table:
        fdr = table["fdr_bh"].to_numpy(dtype=float, copy=False)
        significant_mask = np.isfinite(fdr) & (fdr <= fdr_alpha)

    significant_difference = np.where(significant_mask, difference, 0.0).astype(np.float32, copy=False)

    table["mean_difference"] = difference
    table["significant"] = significant_mask
    table["significant_mean_difference"] = significant_difference
    table["absolute_significant_mean_difference"] = np.abs(significant_difference)

    return DifferentialSignature(
        table=table,
        positive_mean=positive_mean,
        negative_mean=negative_mean,
        difference=difference.astype(np.float32, copy=False),
        significant_difference=significant_difference,
        n_significant=int(significant_mask.sum()),
        fdr_alpha=fdr_alpha,
    )


def run_probe(
    X: sp.csr_matrix,
    y: np.ndarray,
    output_dir: Path,
    top_n_features: int,
) -> ProbeResult:
    y = np.asarray(y, dtype=np.int8)
    classes = np.unique(y)
    coefficients_path = output_dir / "probe_coefficients.csv"

    if classes.size < 2:
        empty = pd.DataFrame(columns=["feature_idx", "coefficient"])
        metrics = pd.DataFrame(
            [
                {
                    "status": "skipped",
                    "reason": "Only one class was present after alignment; probe training requires both classes.",
                }
            ]
        )
        metrics.to_csv(output_dir / "probe_metrics.csv", index=False)
        empty.to_csv(coefficients_path, index=False)
        empty.to_csv(output_dir / "top_positive_features.csv", index=False)
        empty.to_csv(output_dir / "top_negative_features.csv", index=False)
        return ProbeResult(
            metrics=metrics,
            coefficients=empty,
            top_positive=empty,
            top_negative=empty,
            status="skipped",
            message=metrics.loc[0, "reason"],
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    probe = make_pipeline(
        MaxAbsScaler(),
        LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
            random_state=42,
        ),
    )
    probe.fit(X_train, y_train)
    probabilities = probe.predict_proba(X_test)[:, 1]
    predictions = probe.predict(X_test)

    metrics = pd.DataFrame(
        [
            {
                "status": "ok",
                "n_train": X_train.shape[0],
                "n_test": X_test.shape[0],
                "auroc": roc_auc_score(y_test, probabilities),
                "auprc": average_precision_score(y_test, probabilities),
                "accuracy": accuracy_score(y_test, predictions),
                "f1": f1_score(y_test, predictions),
            }
        ]
    )
    metrics.to_csv(output_dir / "probe_metrics.csv", index=False)

    classifier = probe.named_steps["logisticregression"]
    coefficients = pd.DataFrame(
        {
            "feature_idx": np.arange(X.shape[1], dtype=int),
            "coefficient": classifier.coef_.ravel(),
        }
    ).sort_values("coefficient", ascending=False)
    coefficients.to_csv(coefficients_path, index=False)

    top_positive = coefficients.head(top_n_features).copy()
    top_negative = coefficients.tail(top_n_features).sort_values("coefficient").copy()
    top_positive.to_csv(output_dir / "top_positive_features.csv", index=False)
    top_negative.to_csv(output_dir / "top_negative_features.csv", index=False)

    return ProbeResult(
        metrics=metrics,
        coefficients=coefficients,
        top_positive=top_positive,
        top_negative=top_negative,
        status="ok",
        message="probe completed",
    )


def write_differential_signature(
    signature: DifferentialSignature,
    output_dir: Path,
) -> None:
    signature.table.to_csv(output_dir / "differential_feature_signature.csv", index=False)
    np.save(output_dir / "mean_vector_positive.npy", signature.positive_mean)
    np.save(output_dir / "mean_vector_negative.npy", signature.negative_mean)
    np.save(output_dir / "mean_vector_difference.npy", signature.difference)
    np.save(output_dir / "mean_vector_difference_significant.npy", signature.significant_difference)

    top_positive = (
        signature.table[signature.table["significant_mean_difference"] > 0]
        .sort_values("significant_mean_difference", ascending=False)
        .head(100)
    )
    top_negative = (
        signature.table[signature.table["significant_mean_difference"] < 0]
        .sort_values("significant_mean_difference", ascending=True)
        .head(100)
    )
    top_positive.to_csv(output_dir / "differential_signature_top_positive.csv", index=False)
    top_negative.to_csv(output_dir / "differential_signature_top_negative.csv", index=False)


def _top_feature_candidates(
    feature_stats: pd.DataFrame,
    probe_result: ProbeResult,
    differential_signature: DifferentialSignature,
    top_n_features: int,
) -> list[int]:
    selected: list[int] = []

    if probe_result.status == "ok":
        selected.extend(probe_result.top_positive["feature_idx"].tolist())
        selected.extend(probe_result.top_negative["feature_idx"].tolist())

    if "fdr_bh" in feature_stats and feature_stats["fdr_bh"].notna().any():
        ranked = feature_stats.sort_values(["fdr_bh", "log_odds_ratio"], ascending=[True, False])
        selected.extend(ranked.head(top_n_features)["feature_idx"].tolist())

    if "absolute_significant_mean_difference" in differential_signature.table:
        ranked = differential_signature.table.sort_values(
            "absolute_significant_mean_difference",
            ascending=False,
        )
        selected.extend(ranked.head(top_n_features)["feature_idx"].tolist())

    if not selected:
        ranked = feature_stats.sort_values(["mean_activation_all", "activation_rate"], ascending=[False, False])
        selected.extend(ranked.head(top_n_features)["feature_idx"].tolist())

    seen: set[int] = set()
    deduped: list[int] = []
    for feature_idx in selected:
        feature_idx = int(feature_idx)
        if feature_idx not in seen:
            deduped.append(feature_idx)
            seen.add(feature_idx)
    return deduped


def get_top_examples(
    X: sp.csr_matrix,
    aligned_df: pd.DataFrame,
    feature_indices: list[int],
    top_k: int,
) -> pd.DataFrame:
    matrix_csc = X.tocsc()
    rows: list[dict[str, Any]] = []

    for feature_idx in feature_indices:
        column = matrix_csc.getcol(feature_idx)
        if column.nnz == 0:
            continue
        values = column.data
        row_indices = column.indices
        order = np.argsort(values)[::-1][:top_k]
        for rank, pos in enumerate(order, start=1):
            row_idx = int(row_indices[pos])
            record = aligned_df.iloc[row_idx]
            rows.append(
                {
                    "feature_idx": int(feature_idx),
                    "rank_within_feature": rank,
                    "sequence_id": record["sequence_id"],
                    "sequence": record["sequence"],
                    "label_raw": record["label_raw"],
                    "y": int(record["y"]),
                    "activation_value": float(values[pos]),
                    "metadata_json": record["metadata_json"],
                }
            )

    examples = pd.DataFrame(rows)
    return examples.sort_values(["feature_idx", "rank_within_feature"]).reset_index(drop=True)


def _all_kmers(sequence: str, k: int) -> set[str]:
    if len(sequence) < k:
        return set()
    return {sequence[i : i + k] for i in range(len(sequence) - k + 1)}


def _kmer_entropy(kmer: str) -> float:
    counts = Counter(kmer)
    length = len(kmer)
    entropy = 0.0
    for count in counts.values():
        probability = count / length
        entropy -= probability * math.log2(probability)
    return entropy


def _has_short_repeat_unit(kmer: str) -> bool:
    for unit_len in (1, 2, 3):
        if len(kmer) % unit_len != 0:
            continue
        unit = kmer[:unit_len]
        if unit * (len(kmer) // unit_len) == kmer:
            return True
    return False


def _is_informative_kmer(
    kmer: str,
    min_length: int,
    min_unique_bases: int,
    min_entropy: float,
) -> bool:
    if len(kmer) < min_length:
        return False
    if len(set(kmer)) < min_unique_bases:
        return False
    if _has_short_repeat_unit(kmer):
        return False
    if _kmer_entropy(kmer) < min_entropy:
        return False
    return True


def run_kmer_enrichment(
    examples_df: pd.DataFrame,
    aligned_df: pd.DataFrame,
    k_values: tuple[int, ...] = (5, 6),
    min_kmer_length: int = 5,
    min_unique_bases: int = 3,
    min_entropy: float = 1.75,
    top_kmers_per_feature: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for feature_idx, group in examples_df.groupby("feature_idx"):
        selected_ids = set(group["sequence_id"])
        selected_sequences = aligned_df[aligned_df["sequence_id"].isin(selected_ids)]["sequence"].tolist()
        background_sequences = aligned_df[~aligned_df["sequence_id"].isin(selected_ids)]["sequence"].tolist()
        if not selected_sequences or not background_sequences:
            continue

        for k in k_values:
            selected_counts: dict[str, int] = {}
            background_counts: dict[str, int] = {}

            for sequence in selected_sequences:
                for kmer in _all_kmers(sequence, k):
                    selected_counts[kmer] = selected_counts.get(kmer, 0) + 1
            for sequence in background_sequences:
                for kmer in _all_kmers(sequence, k):
                    background_counts[kmer] = background_counts.get(kmer, 0) + 1

            enriched_rows: list[dict[str, Any]] = []
            n_selected = len(selected_sequences)
            n_background = len(background_sequences)
            for kmer, selected_present in selected_counts.items():
                if not _is_informative_kmer(
                    kmer,
                    min_length=min_kmer_length,
                    min_unique_bases=min_unique_bases,
                    min_entropy=min_entropy,
                ):
                    continue
                background_present = background_counts.get(kmer, 0)
                table = np.array(
                    [
                        [selected_present, n_selected - selected_present],
                        [background_present, n_background - background_present],
                    ],
                    dtype=int,
                )
                _, p_value = fisher_exact(table, alternative="greater")
                enrichment_ratio = ((selected_present + 0.5) / (n_selected + 1.0)) / (
                    (background_present + 0.5) / (n_background + 1.0)
                )
                enriched_rows.append(
                    {
                        "feature_idx": int(feature_idx),
                        "k": k,
                        "kmer": kmer,
                        "selected_count": selected_present,
                        "background_count": background_present,
                        "selected_fraction": selected_present / n_selected,
                        "background_fraction": background_present / n_background,
                        "enrichment_ratio": enrichment_ratio,
                        "p_value": p_value,
                    }
                )

            if not enriched_rows:
                continue

            kmer_df = pd.DataFrame(enriched_rows).sort_values(
                ["p_value", "enrichment_ratio", "selected_count"],
                ascending=[True, False, False],
            )
            kmer_df["fdr_bh"] = _bh_fdr(kmer_df["p_value"].to_numpy())
            rows.extend(kmer_df.head(top_kmers_per_feature).to_dict("records"))

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature_idx",
                "k",
                "kmer",
                "selected_count",
                "background_count",
                "selected_fraction",
                "background_fraction",
                "enrichment_ratio",
                "p_value",
                "fdr_bh",
            ]
        )

    return pd.DataFrame(rows).sort_values(["feature_idx", "k", "p_value", "enrichment_ratio"])


def _project_for_plot(X: sp.csr_matrix, n_components: int = 2) -> np.ndarray:
    if X.shape[0] < 2:
        return np.zeros((X.shape[0], n_components), dtype=np.float32)
    if X.shape[1] > 256:
        reducer = TruncatedSVD(n_components=min(50, X.shape[1] - 1), random_state=42)
        reduced = reducer.fit_transform(X)
        pca = PCA(n_components=n_components, random_state=42)
        return pca.fit_transform(reduced)
    return PCA(n_components=n_components, random_state=42).fit_transform(X.toarray())


def make_plots(
    X: sp.csr_matrix,
    y: np.ndarray,
    feature_stats: pd.DataFrame,
    probe_result: ProbeResult,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y, dtype=int)

    plt.figure(figsize=(8, 4.5))
    plt.hist(feature_stats["activation_rate"], bins=50, color="#1f77b4", edgecolor="white")
    plt.xlabel("Activation rate")
    plt.ylabel("Feature count")
    plt.title("SAE Feature Activation Rate Histogram")
    plt.tight_layout()
    plt.savefig(output_dir / "activation_rate_histogram.png", dpi=180)
    plt.close()

    projected = _project_for_plot(X)
    plt.figure(figsize=(6.5, 5.5))
    scatter = plt.scatter(projected[:, 0], projected[:, 1], c=y, s=12, cmap="coolwarm", alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Projection of SAE Features")
    if np.unique(y).size > 1:
        plt.colorbar(scatter, label="class-1 label")
    plt.tight_layout()
    plt.savefig(output_dir / "pca_projection.png", dpi=180)
    plt.close()

    tsne_sample = min(5000, X.shape[0])
    if tsne_sample >= 10:
        sample_idx = np.linspace(0, X.shape[0] - 1, num=tsne_sample, dtype=int)
        sample_projection_input = X[sample_idx]
        if sample_projection_input.shape[1] > 256:
            sample_projection_input = TruncatedSVD(
                n_components=min(50, sample_projection_input.shape[1] - 1),
                random_state=42,
            ).fit_transform(sample_projection_input)
        else:
            sample_projection_input = sample_projection_input.toarray()

        perplexity = max(5, min(30, tsne_sample // 10))
        embedded = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=42,
        ).fit_transform(sample_projection_input)
        plt.figure(figsize=(6.5, 5.5))
        scatter = plt.scatter(
            embedded[:, 0],
            embedded[:, 1],
            c=y[sample_idx],
            s=12,
            cmap="coolwarm",
            alpha=0.7,
        )
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title("t-SNE Projection of SAE Features")
        if np.unique(y).size > 1:
            plt.colorbar(scatter, label="class-1 label")
        plt.tight_layout()
        plt.savefig(output_dir / "tsne_projection.png", dpi=180)
        plt.close()

    try:
        import umap

        umap_input = X
        if X.shape[1] > 256:
            umap_input = TruncatedSVD(n_components=min(50, X.shape[1] - 1), random_state=42).fit_transform(X)
        else:
            umap_input = X.toarray()
        embedding = umap.UMAP(random_state=42).fit_transform(umap_input)
        plt.figure(figsize=(6.5, 5.5))
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=y, s=12, cmap="coolwarm", alpha=0.7)
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.title("UMAP Projection of SAE Features")
        if np.unique(y).size > 1:
            plt.colorbar(scatter, label="class-1 label")
        plt.tight_layout()
        plt.savefig(output_dir / "umap_projection.png", dpi=180)
        plt.close()
    except Exception:
        pass

    if feature_stats["fdr_bh"].notna().any():
        volcano = feature_stats.copy()
        volcano["neg_log10_p"] = -np.log10(np.clip(volcano["p_value"], 1e-300, 1.0))
        plt.figure(figsize=(7, 5.5))
        plt.scatter(volcano["log_odds_ratio"], volcano["neg_log10_p"], s=12, alpha=0.6, color="#444444")
        plt.xlabel("Log odds ratio")
        plt.ylabel("-log10(p-value)")
        plt.title("Volcano Plot of Feature Enrichment")
        plt.tight_layout()
        plt.savefig(output_dir / "volcano_plot.png", dpi=180)
        plt.close()

    if probe_result.status == "ok":
        bar_df = pd.concat(
            [
                probe_result.top_positive.assign(direction="Positive"),
                probe_result.top_negative.assign(direction="Negative"),
            ],
            ignore_index=True,
        )
        labels = [f"{row.direction} #{int(row.feature_idx)}" for row in bar_df.itertuples()]
        plt.figure(figsize=(10, 6))
        plt.barh(labels, bar_df["coefficient"], color=["#b22222" if x > 0 else "#1f4f99" for x in bar_df["coefficient"]])
        plt.xlabel("Probe coefficient")
        plt.title("Top Class-Associated SAE Features")
        plt.tight_layout()
        plt.savefig(output_dir / "top_class_features.png", dpi=180)
        plt.close()


def write_report(
    aligned_df: pd.DataFrame,
    feature_stats: pd.DataFrame,
    probe_result: ProbeResult,
    examples_df: pd.DataFrame,
    kmer_df: pd.DataFrame,
    output_dir: Path,
    activation_bundle: ActivationBundle,
    label_field: str,
    positive_label: str,
    k_values: tuple[int, ...],
    min_kmer_length: int,
    min_unique_bases: int,
    min_entropy: float,
    differential_signature: DifferentialSignature,
) -> None:
    label_counts = aligned_df["label_raw"].value_counts().sort_index()
    top_overall = feature_stats.sort_values(["mean_activation_all", "activation_rate"], ascending=[False, False]).head(10)
    report_lines = [
        "# SAE Feature Analysis Report",
        "",
        "## Dataset",
        f"- Activation source: `{activation_bundle.source_path}` ({activation_bundle.description})",
        f"- Dataset rows aligned: {len(aligned_df):,}",
        f"- SAE shape: {activation_bundle.matrix.shape[0]:,} sequences x {activation_bundle.matrix.shape[1]:,} features",
        f"- Label field: `{label_field}`",
        f"- Positive label: `{positive_label}`",
        "- Label balance:",
    ]
    report_lines.extend([f"  - `{label}`: {count:,}" for label, count in label_counts.items()])

    report_lines.extend(
        [
            "",
            "## Strongest SAE Features",
            "Top features by overall mean activation:",
        ]
    )
    for row in top_overall.itertuples():
        report_lines.append(
            f"- Feature {int(row.feature_idx)}: mean={row.mean_activation_all:.4f}, activation_rate={row.activation_rate:.4f}"
        )

    report_lines.extend(
        [
            "",
            "## Differential Signature",
            f"- Computed class-mean vectors for `{positive_label}` and its complement, then subtracted them.",
            f"- Features retained after significance masking (Benjamini-Hochberg FDR <= {differential_signature.fdr_alpha:.3f}): {differential_signature.n_significant:,}",
            "- Saved vectors:",
            "  - `mean_vector_positive.npy`",
            "  - `mean_vector_negative.npy`",
            "  - `mean_vector_difference.npy`",
            "  - `mean_vector_difference_significant.npy`",
        ]
    )

    top_positive_signature = (
        differential_signature.table[differential_signature.table["significant_mean_difference"] > 0]
        .sort_values("significant_mean_difference", ascending=False)
        .head(10)
    )
    top_negative_signature = (
        differential_signature.table[differential_signature.table["significant_mean_difference"] < 0]
        .sort_values("significant_mean_difference", ascending=True)
        .head(10)
    )
    if not top_positive_signature.empty:
        report_lines.append("- Largest positive significant mean differences:")
        for row in top_positive_signature.itertuples():
            report_lines.append(
                f"- Feature {int(row.feature_idx)}: delta={row.significant_mean_difference:.5f}, "
                f"fdr={row.fdr_bh:.2e}"
            )
    if not top_negative_signature.empty:
        report_lines.append("- Largest negative significant mean differences:")
        for row in top_negative_signature.itertuples():
            report_lines.append(
                f"- Feature {int(row.feature_idx)}: delta={row.significant_mean_difference:.5f}, "
                f"fdr={row.fdr_bh:.2e}"
            )

    if probe_result.status == "ok":
        metrics_row = probe_result.metrics.iloc[0]
        report_lines.extend(
            [
                "",
                "## Probe Performance",
                f"- AUROC: {metrics_row['auroc']:.4f}",
                f"- AUPRC: {metrics_row['auprc']:.4f}",
                f"- Accuracy: {metrics_row['accuracy']:.4f}",
                f"- F1: {metrics_row['f1']:.4f}",
                "",
                "Top positive features:",
            ]
        )
        for row in probe_result.top_positive.itertuples():
            report_lines.append(f"- Feature {int(row.feature_idx)}: coefficient={row.coefficient:.4f}")
        report_lines.append("")
        report_lines.append("Top negative features:")
        for row in probe_result.top_negative.itertuples():
            report_lines.append(f"- Feature {int(row.feature_idx)}: coefficient={row.coefficient:.4f}")
    else:
        report_lines.extend(
            [
                "",
                "## Probe Performance",
                f"- Skipped: {probe_result.message}",
            ]
        )

    report_lines.extend(
        [
            "",
            "## Top Examples",
            f"- Saved {len(examples_df):,} feature-sequence examples to `top_feature_examples.csv`.",
        ]
    )
    if not kmer_df.empty:
        report_lines.append(
            f"- Candidate k-mers from top activating sequences "
            f"(k in {list(k_values)}, min length {min_kmer_length}, "
            f"min unique bases {min_unique_bases}, min entropy {min_entropy:.2f}):"
        )
        for row in kmer_df.head(10).itertuples():
            report_lines.append(
                f"- Feature {int(row.feature_idx)} k={int(row.k)} motif `{row.kmer}` "
                f"(enrichment={row.enrichment_ratio:.2f}, p={row.p_value:.2e})"
            )
    else:
        report_lines.append(
            f"- No informative k-mers passed the filters "
            f"(k in {list(k_values)}, min length {min_kmer_length}, "
            f"min unique bases {min_unique_bases}, min entropy {min_entropy:.2f})."
        )

    report_lines.extend(
        [
            "",
            "## Candidate Biological Interpretations",
            "- Features that are both class-enriched and strong in the linear probe are the best initial candidates for human-infecting viral signatures.",
            "- Top-activating sequences and enriched k-mers provide the first pass at motif-level interpretation, but they should be validated against held-out data and known viral sequence motifs.",
            "- If only one class is present, this report is limited to within-class latent summaries; class-specific claims require negative-class activations encoded with the same SAE.",
        ]
    )

    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SAE sequence-level activations against binary labels")
    parser.add_argument("--dataset_jsonl", required=True, help="Path to dataset JSONL")
    parser.add_argument("--activation_path", required=True, help="Activation file or directory")
    parser.add_argument("--output_dir", required=True, help="Directory for CSVs, plots, and report")
    parser.add_argument("--label_field", default="metadata.class", help="Dot path to label field")
    parser.add_argument("--positive_label", default="class-1", help="Positive label value")
    parser.add_argument("--top_features", type=int, default=25, help="Number of top features to report")
    parser.add_argument("--top_examples", type=int, default=20, help="Number of top examples per feature")
    parser.add_argument(
        "--kmer_lengths",
        default="5,6",
        help="Comma-separated k-mer lengths to analyze. Defaults to 5,6 to avoid low-information short motifs.",
    )
    parser.add_argument(
        "--min_kmer_unique_bases",
        type=int,
        default=3,
        help="Minimum number of distinct bases required for a k-mer to be reported.",
    )
    parser.add_argument(
        "--min_kmer_entropy",
        type=float,
        default=1.75,
        help="Minimum Shannon entropy required for a k-mer to be reported.",
    )
    parser.add_argument(
        "--fdr_alpha",
        type=float,
        default=0.05,
        help="Benjamini-Hochberg FDR threshold used to keep or zero differential-signature components.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_df = load_dataset(args.dataset_jsonl, args.label_field, args.positive_label)
    activation_bundle = load_activations(args.activation_path)
    aligned_df, X = align_records(dataset_df, activation_bundle)
    y = aligned_df["y"].to_numpy(dtype=np.int8)

    feature_stats = compute_feature_stats(X, y)
    feature_stats.to_csv(output_dir / "feature_stats.csv", index=False)
    differential_signature = compute_differential_signature(feature_stats, fdr_alpha=args.fdr_alpha)
    write_differential_signature(differential_signature, output_dir)

    probe_result = run_probe(X, y, output_dir, args.top_features)
    selected_features = _top_feature_candidates(
        feature_stats,
        probe_result,
        differential_signature,
        args.top_features,
    )

    examples_df = get_top_examples(X, aligned_df, selected_features, args.top_examples)
    examples_df.to_csv(output_dir / "top_feature_examples.csv", index=False)

    k_values = tuple(
        sorted(
            {
                int(value.strip())
                for value in args.kmer_lengths.split(",")
                if value.strip()
            }
        )
    )
    if not k_values:
        raise ValueError("--kmer_lengths must specify at least one integer k")
    min_kmer_length = min(k_values)

    kmer_df = run_kmer_enrichment(
        examples_df,
        aligned_df,
        k_values=k_values,
        min_kmer_length=min_kmer_length,
        min_unique_bases=args.min_kmer_unique_bases,
        min_entropy=args.min_kmer_entropy,
    )
    kmer_df.to_csv(output_dir / "kmer_enrichment.csv", index=False)

    make_plots(X, y, feature_stats, probe_result, output_dir)
    write_report(
        aligned_df=aligned_df,
        feature_stats=feature_stats,
        probe_result=probe_result,
        examples_df=examples_df,
        kmer_df=kmer_df,
        output_dir=output_dir,
        activation_bundle=activation_bundle,
        label_field=args.label_field,
        positive_label=args.positive_label,
        k_values=k_values,
        min_kmer_length=min_kmer_length,
        min_unique_bases=args.min_kmer_unique_bases,
        min_entropy=args.min_kmer_entropy,
        differential_signature=differential_signature,
    )

    print(f"Wrote analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
