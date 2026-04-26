"""
MetaGeniuses API backend.

Serves experiment results and feature explorer data.
Reads from results/ directories if real data exists, otherwise returns dummy data.

To add real data: just drop JSON/CSV files into the appropriate results/<experiment>/ directory.
The backend will pick them up automatically — no frontend changes needed.
"""

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dummy_data import (
    generate_features,
    generate_experiment1,
    generate_experiment2,
    generate_experiment3,
    generate_experiment4,
    generate_experiment5,
)

app = FastAPI(title="MetaGeniuses API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _try_load_json(path: Path):
    """Load JSON file if it exists, return None otherwise."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Feature Explorer (existing UI)
# ---------------------------------------------------------------------------

@app.get("/api/features")
def get_features():
    """SAE feature list for the explorer sidebar + detail panels."""
    real = _try_load_json(RESULTS_DIR / "features" / "features.json")
    if real is not None:
        return real
    return generate_features()


@app.get("/api/features/{feature_id}")
def get_feature(feature_id: int):
    """Single feature with full detail (histogram, top sequences, taxa)."""
    real = _try_load_json(RESULTS_DIR / "features" / f"feature_{feature_id}.json")
    if real is not None:
        return real
    # Fall back to finding it in the full dummy set
    for f in generate_features():
        if f["id"] == feature_id:
            return f
    return {"error": "feature not found"}


# ---------------------------------------------------------------------------
# Experiment 1: Organism-Specific Pathogen Detectors
# ---------------------------------------------------------------------------

@app.get("/api/experiments/1")
def experiment1():
    """Volcano plot data, top organism detectors, enrichment stats."""
    real = _try_load_json(RESULTS_DIR / "organism_detectors" / "api_results.json")
    if real is not None:
        return real
    return generate_experiment1()


# ---------------------------------------------------------------------------
# Experiment 2: Linear Probe
# ---------------------------------------------------------------------------

@app.get("/api/experiments/2")
def experiment2():
    """Probe summary, ROC curve, coefficient distribution, top latents."""
    real = _try_load_json(RESULTS_DIR / "linear_probe_pathogen" / "api_results.json")
    if real is not None:
        return real
    return generate_experiment2()


# ---------------------------------------------------------------------------
# Experiment 3: SAE Health Check
# ---------------------------------------------------------------------------

@app.get("/api/experiments/3")
def experiment3():
    """Dead/alive census, activation distributions, sparsity stats."""
    real = _try_load_json(RESULTS_DIR / "sae_health_check" / "api_results.json")
    if real is not None:
        return real
    return generate_experiment3()


# ---------------------------------------------------------------------------
# Experiment 4: Sequence UMAP
# ---------------------------------------------------------------------------

@app.get("/api/experiments/4")
def experiment4():
    """UMAP coordinates colored by pathogen label, PCA variance."""
    real = _try_load_json(RESULTS_DIR / "sequence_umap" / "api_results.json")
    if real is not None:
        return real
    return generate_experiment4()


# ---------------------------------------------------------------------------
# Experiment 5: Feature Clustering
# ---------------------------------------------------------------------------

@app.get("/api/experiments/5")
def experiment5():
    """Latent UMAP colored by cluster, enrichment overlay, cluster summary."""
    real = _try_load_json(RESULTS_DIR / "feature_clustering" / "api_results.json")
    if real is not None:
        return real
    return generate_experiment5()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
