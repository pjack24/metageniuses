# RunPod Setup For MetaGene Extraction

This runbook uses the repo's production preference config:

- `configs/extraction/metagene-cloud-prod.json`
- Current preference: `4 layers`, `20,000 sequences`, input `data/human_virus_class1.jsonl`

Outputs are isolated to:

- `/root/results/cloud-extraction/`

## 1. Create RunPod Pod

1. Create a new Pod (recommended: 80GB+ VRAM GPU for 7B model comfort).
2. Use a PyTorch image (Python 3.10+).
3. Attach enough disk for:
   - HF cache (~20+ GB for METAGENE-1)
   - activation outputs (depends on layers/dtype; plan 100+ GB headroom for larger jobs)
4. Expose SSH or Jupyter terminal.
5. Before launch, confirm `/root/results/cloud-extraction` is on the attached pod disk, not a slow network mount.

## 2. Connect And Prepare Repo

```bash
git clone <your_repo_url>
cd metageniuses
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers huggingface_hub safetensors accelerate
```

## 3. Authenticate Hugging Face

```bash
huggingface-cli login
```

Or set an env var:

```bash
export HF_TOKEN=<your_token>
```

## 4. Verify Config Before Launch

Check:

1. `input.path` points to the intended 20k file.
2. `layer_selection.last_n_layers` is `4`.
3. `runtime.max_reads` is `20000`.
4. `runtime.output_root` is `/root/results/cloud-extraction`.
5. `runtime.batch_size` is `20`.
6. `runtime.initial_max_batch_size` is `40` and `runtime.max_batch_size` is `60`.
7. `runtime.release_to_max_after_sequences` is set (currently `5000`).

## 5. Run Extraction

```bash
PYTHONPATH=src python -m metageniuses.extraction.cli \
  --config configs/extraction/metagene-cloud-prod.json \
  --adapter transformers
```

This writes to `/root/results/cloud-extraction/<runtime.run_id>/`. With the checked-in
cloud config, `runtime.run_id` is fixed, so rerunning without `--resume` will stop
instead of overwriting an existing run.

## 5b. Resume After Disconnect

If your SSH/Jupyter session drops, relaunch with the same config and `--resume`:

```bash
PYTHONPATH=src python -m metageniuses.extraction.cli \
  --config configs/extraction/metagene-cloud-prod.json \
  --adapter transformers \
  --resume
```

Requirements for resume:

1. Use the same explicit `runtime.run_id` from the first launch.
2. Keep the same `runtime.output_root`.
3. Keep the same selected layers/model/input ordering.

## 6. Validate Artifacts

Expected run folder:

- `/root/results/cloud-extraction/<runtime.run_id>/`

Expected files:

1. `manifest.json`
2. `sequences.jsonl`
3. `activations/layer_XX/shard_*.f32`
4. `activations/layer_XX/shard_*.jsonl`

Quick check:

```bash
python - <<'PY'
import json
m=json.load(open('/root/results/cloud-extraction/<run_id>/manifest.json'))
print('layers',m['layer_selection']['selected_transformer_layers'])
print('stats',m['stats'])
PY
```

SAE format sanity check:

```bash
PYTHONPATH=src python - <<'PY'
from metageniuses.extraction.contracts import iter_layer_batches
run_root='/root/results/cloud-extraction/<run_id>'
layer=32
vectors,meta=next(iter(iter_layer_batches(run_root, layer, batch_size=8)))
print('batch_rows',len(vectors))
print('vector_dim',len(vectors[0]))
print('sample_meta_keys',sorted(meta[0].keys()))
PY
```

## 7. Download Results

Use `rsync`, `scp`, or object storage sync from:

- `/root/results/cloud-extraction/<runtime.run_id>/`

Keep `data/` untouched as source/test data.
