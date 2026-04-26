# Experiment 1: Organism-Specific Pathogen Detectors

Find SAE latents that fire specifically on pathogen sequences, then use BLAST to identify *which organism* each latent is detecting. This is the main result of the paper.

## Motivation

MetaGene-1 can detect pathogens (92.96 MCC). But *why*? We show it has internally learned organism-level detectors — individual SAE features that fire on Influenza A, or SARS-CoV-2, or norovirus — without ever being trained with organism labels.

This is the analog of InterProt finding protein-family-specific features (Section 4.2, Figure 3a) and SURF finding PBD-family-specific features. But for pathogen species in metagenomic data.

## Prior results that set this up

- **Exp 2 (linear probe)**: 94.6% accuracy, 0.892 MCC — proves pathogen signal exists in SAE features
- **Exp 4 (sequence UMAP)**: clear separation + 49 sub-clusters within pathogen class — sub-clusters may correspond to different organisms
- **Exp 5 (feature clustering)**: 12,894 pathogen-enriched latents (log2 OR > 1) — pathogen features cluster spatially in latent UMAP
- **Exp 2 insight**: top probe latents have enrichment ~1.0 (fire on everything, differ by magnitude). Enrichment analysis finds a *different* set of latents — ones with skewed firing frequency. Both are complementary.

## Data

- `data/sae_model/features.npy` — (20,000 x 32,768) float32, sequence-level SAE activations
- `data/sae_model/sequence_ids.json` — JSON array, maps row index to sequence_id string
- `data/human_virus_class1_labeled.jsonl` — one JSON object per line with fields:
  - `sequence_id`: string like "human_virus_class1_0"
  - `source`: string "0" (non-pathogen) or "1" (pathogen), 10k each
  - `class`: string "class-1" (all rows)
  - `sequence`: nucleotide string (the actual DNA/RNA read, ~100-300 bp)

## Implementation

Single script: `experiments/organism_detectors.py`

Output directory: `results/organism_detectors/` (create with `mkdir -p`)

### Part A: Enrichment scan (~2 min, local)

For each of 32,768 latents, compute three enrichment metrics against the binary `source` label.

**A1. Fisher's exact test (binary: active vs inactive)**

```python
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# For each latent i:
col = features[:, i]
active = col > 0
table = [
    [active[pathogen].sum(), active[~pathogen].sum()],      # active: pathogen, non-pathogen
    [(~active[pathogen]).sum(), (~active[~pathogen]).sum()],  # inactive: pathogen, non-pathogen
]
odds_ratio, p_value = fisher_exact(table)
```

- Collect all 32,768 p-values
- FDR-correct: `reject, fdr_pvals, _, _ = multipletests(p_values, alpha=0.01, method='fdr_bh')`
- A latent is "pathogen-enriched" if `fdr_pval < 0.01` AND `odds_ratio > 1`
- A latent is "non-pathogen-enriched" if `fdr_pval < 0.01` AND `odds_ratio < 1`

**A2. Log-fold-change + Wilcoxon rank-sum (continuous activation values)**

```python
from scipy.stats import mannwhitneyu

# For each latent i:
col = features[:, i]
mean_p = col[pathogen].mean()
mean_np = col[~pathogen].mean()
eps = 1e-10
log2fc = np.log2((mean_p + eps) / (mean_np + eps))
stat, wilcox_p = mannwhitneyu(col[pathogen], col[~pathogen], alternative='two-sided')
```

- FDR-correct the Wilcoxon p-values separately
- This captures magnitude differences that Fisher's misses (Fisher only sees active/inactive)

**A3. InterProt-style F1 sweep**

```python
from sklearn.metrics import f1_score

# For each latent i:
col = features[:, i]
max_val = col.max()
if max_val == 0:
    best_f1 = 0.0
else:
    best_f1 = 0.0
    for threshold_frac in np.arange(0.05, 1.0, 0.05):
        threshold = threshold_frac * max_val
        pred = (col >= threshold).astype(int)
        f1 = f1_score(pathogen.astype(int), pred, zero_division=0)
        best_f1 = max(best_f1, f1)
```

- If best F1 > 0.7, call it "pathogen-specific" (matching InterProt Appendix A.6)
- This is the most stringent criterion — single latent must be a decent classifier on its own

**Output A**: `enrichment_results.csv` with columns:
```
latent_id, fisher_or, fisher_p, fisher_fdr, log2fc, wilcoxon_p, wilcoxon_fdr, best_f1,
is_pathogen_enriched, is_nonpathogen_enriched, is_pathogen_specific,
activation_count_pathogen, activation_count_nonpathogen,
mean_activation_pathogen, mean_activation_nonpathogen
```

Print summary: how many latents pass each criterion.

### Part B: Sequence retrieval (~10 sec, local)

Select top latents for BLAST. Use a tiered approach:
1. First priority: latents with `is_pathogen_specific == True` (F1 > 0.7) — these are the strongest
2. Second priority: latents with `is_pathogen_enriched == True`, sorted by Fisher odds ratio descending
3. Cap at 50 latents total

For each selected latent:
- Take the 10,000 pathogen sequences (source=1)
- Sort by activation value for that latent, descending
- Take the top 10
- Look up their nucleotide sequences from the labeled JSONL

**Output B**: `top_sequences_per_latent.json` — structure:
```json
{
  "7241": {
    "fisher_or": 4.2,
    "fisher_fdr": 1.2e-15,
    "log2fc": 2.1,
    "best_f1": 0.73,
    "top_sequences": [
      {
        "sequence_id": "human_virus_class1_4523",
        "activation": 0.234,
        "sequence": "ATGCGTACC..."
      },
      ...
    ]
  },
  ...
}
```

### Part C: BLAST (~15-30 min, NCBI API)

Use the NCBI BLAST REST API. The API works in two phases: submit a search, then poll for results.

**Submit a search:**
```
POST https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi
Content-Type: application/x-www-form-urlencoded

CMD=Put&PROGRAM=blastn&DATABASE=nt&QUERY=ATGCGTACC...
```

Response contains a Request ID (RID) like `RID = ABCDE12345`.

**Poll for results:**
```
GET https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD=Get&FORMAT_TYPE=JSON2&RID=ABCDE12345
```

Response contains `Status`: `WAITING`, `READY`, or `UNKNOWN`. Poll every 15 seconds until `READY`.

**Rate limiting:**
- NCBI asks for max ~3 requests/second
- Include `email` and `tool` parameters in requests (good citizenship): `TOOL=metageniuses&EMAIL=mannat@columbia.edu`
- Sleep 0.5s between submissions
- Sleep 15s between poll attempts
- If we get 429 or 500 errors, back off exponentially

**Batching strategy:**
- Submit all ~500 sequences upfront, collect RIDs
- Poll all RIDs in round-robin until all complete
- This parallelizes the BLAST computation on NCBI's side

**Parsing results:**
For each BLAST result, extract from the top hit:
- `hit_organism`: organism name (e.g., "Influenza A virus")
- `hit_accession`: GenBank accession
- `hit_description`: full hit description
- `percent_identity`: alignment identity %
- `e_value`: expect value
- `query_coverage`: how much of our sequence aligned
- `bit_score`: alignment score

**Edge cases:**
- No hits found: record `hit_organism = "no_hit"`. Metagenomic reads from novel/poorly-characterized organisms may not match anything in GenBank.
- Very short sequences (<50 bp): may get spurious hits. Record but flag with `short_query = True`.
- "Uncultured" or "environmental sample" hits: record the full description. These are common in metagenomic data and are valid results — they tell us the feature is detecting something real but uncharacterized.
- BLAST timeout: if a job doesn't complete within 10 minutes, mark as `status = "timeout"` and move on.
- Network errors: retry up to 3 times with exponential backoff.

**Checkpoint/resume:**
Save results incrementally to `blast_results_partial.json` after each latent completes. If the script is interrupted, check for existing results and skip already-completed latents on restart.

**Output C**: `blast_results.json` — structure:
```json
{
  "7241": {
    "sequences_submitted": 10,
    "sequences_with_hits": 9,
    "hits": [
      {
        "sequence_id": "human_virus_class1_4523",
        "top_hit": {
          "organism": "Influenza A virus (A/California/07/2009(H1N1))",
          "accession": "CY121680.1",
          "description": "Influenza A virus ... polymerase PB2 ...",
          "percent_identity": 96.3,
          "e_value": 2.1e-45,
          "query_coverage": 98,
          "bit_score": 312
        },
        "status": "hit"
      },
      ...
    ]
  },
  ...
}
```

### Part D: Organism labeling (~1 min, local)

For each latent, tally the organisms from its BLAST hits:

```python
from collections import Counter

organisms = Counter()
for hit in latent_hits:
    if hit["status"] == "hit":
        # Extract genus-level name (first two words of organism)
        org = " ".join(hit["top_hit"]["organism"].split()[:2])
        organisms[org] += 1

dominant_org, count = organisms.most_common(1)[0]
hit_consistency = f"{count}/{len([h for h in latent_hits if h['status'] == 'hit'])}"
```

Labeling rules:
- `count >= 7` out of sequences with hits → label = dominant organism, confidence = "high"
- `count >= 5` → label = dominant organism, confidence = "medium"
- `count < 5` → label = "mixed/unresolved", confidence = "low"
- All hits are "no_hit" → label = "uncharacterized", confidence = "none"
- All hits are "uncultured" or "environmental" → label = "uncharacterized environmental", confidence = "low"

**Output D**: `organism_labels.csv` with columns:
```
latent_id, fisher_or, fisher_fdr, log2fc, best_f1,
dominant_organism, hit_consistency, confidence,
representative_gene, mean_percent_identity, mean_e_value
```

### Part E: Figures

**E1. Volcano plot** (`volcano_plot.png`)
- x-axis: log2FC (from A2)
- y-axis: -log10(Fisher FDR p-value) (from A1)
- Color: gray for non-significant, red for pathogen-enriched (FDR < 0.01, OR > 1), blue for non-pathogen-enriched
- Mark the top 10 organism-labeled latents with text annotations
- Size: (10, 7), dpi 150

**E2. Organism detector bar chart** (`top_organism_detectors.png`)
- Horizontal bar chart of top 15 latents with high-confidence organism labels
- Bar length = Fisher odds ratio
- Bar label = "L{id}: {organism} ({consistency})"
- Color by organism (one color per unique organism)
- Size: (12, 8), dpi 150

**E3. Enrichment distribution** (`enrichment_histogram.png`)
- Histogram of log2(Fisher OR) across all alive latents
- Mark the significance threshold
- Annotate: "X pathogen-enriched, Y non-pathogen-enriched, Z pathogen-specific (F1>0.7)"
- Size: (10, 5), dpi 150

**E4. Organism consistency heatmap** (`organism_heatmap.png`, optional)
- For each labeled latent (rows) x top 5 organisms across all latents (columns)
- Cell = count of BLAST hits to that organism
- Shows whether latents are truly organism-specific or cross-reactive

## Dependencies

```
pip install numpy scipy statsmodels scikit-learn matplotlib requests
```

All data present in repo. NCBI BLAST API is free and public.

## Runtime estimate

| Step | Time | Compute |
|------|------|---------|
| A: Enrichment scan | ~2-3 min | Local (scipy, 32k Fisher tests) |
| B: Sequence retrieval | ~10 sec | Local (numpy indexing) |
| C: BLAST | ~15-30 min | NCBI API (rate-limited, parallelized via batched RIDs) |
| D: Organism labeling | ~10 sec | Local (Counter) |
| E: Figures | ~30 sec | Local (matplotlib) |
| **Total** | **~20-35 min** | **$0** |

## What success looks like

A table of SAE latents with organism labels backed by BLAST evidence:

| Latent | Fisher OR | FDR | F1 | Organism | Consistency | Gene | Identity |
|--------|----------|-----|----|---------:|-------------|------|----------|
| 7241 | 4.2 | 1.2e-15 | 0.73 | Influenza A | 9/10 | polymerase PB2 | 96% |
| 12803 | 3.8 | 3.4e-12 | 0.68 | SARS-CoV-2 | 8/10 | spike protein | 94% |

## What failure looks like

- All BLAST hits are "uncultured" / "environmental sample" → the sequences are too novel for GenBank. Still publishable: "MetaGene-1 has learned features for uncharacterized viral sequences."
- No latents pass F1 > 0.7 → pathogen detection is distributed, not concentrated in individual features. Fall back to enrichment-only analysis (Fisher OR) without the InterProt-style specificity claim.
- BLAST times out or rate-limits us → reduce to top 20 latents x 5 sequences = 100 queries. Should always be feasible.

## What NOT to edit

- `src/metageniuses/sae/analyze.py` — Peyton's pipeline, separate codebase
- `tests/sae/test_analyze.py` — Peyton's tests
- `pyproject.toml` — shared config
