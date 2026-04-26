"""
Dummy data generators for each experiment endpoint.

Each function returns a dict matching the exact schema that the frontend expects.
When real experiment scripts run, they should write an `api_results.json` file
to the appropriate results/ subdirectory with the SAME schema — then the backend
will serve real data instead of these dummies, with zero frontend changes.

Schema contracts are documented inline.
"""

import math
import random

random.seed(42)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_tokens(seq: str, region: tuple[int, int]):
    return [
        {
            "text": ch,
            "activation": (
                round(0.4 + random.random() * 0.6, 3)
                if region[0] <= i < region[1]
                else round(random.random() * 0.08, 3)
            ),
        }
        for i, ch in enumerate(seq)
    ]


def _make_histogram(bins: int = 20):
    vals = [math.exp(-i * 0.3) * (0.6 + random.random() * 0.4) for i in range(bins)]
    mx = max(vals)
    return [round(v / mx, 3) for v in vals]


# ── Feature Explorer ─────────────────────────────────────────────────────────

def generate_features() -> list[dict]:
    """
    Schema per feature:
    {
        id: int,
        label: str,
        description: str,
        layer: int,
        freqActive: float (0-1),
        maxAct: float,
        category: str,          # "Short Motif"|"Domain"|"Point"|"Whole"|"Periodic"
        activationPattern: str, # same as category
        histogram: list[float], # 20 bins, normalized 0-1
        topTaxa: [{name: str, score: float}],
        topSequences: [{
            source: str,
            classLabel: str|null,
            maxAct: float,
            tokens: [{text: str, activation: float}]
        }]
    }
    """
    return [
        {
            "id": 127,
            "label": "Viral capsid motif",
            "description": "Activates on conserved viral capsid protein signatures. Enriched in human-infecting RNA viruses.",
            "layer": 16,
            "freqActive": 0.034,
            "maxAct": 8.42,
            "category": "Short Motif",
            "activationPattern": "Short Motif",
            "histogram": _make_histogram(),
            "topTaxa": [
                {"name": "Influenza A", "score": 0.82},
                {"name": "SARS-CoV-2", "score": 0.71},
                {"name": "Rhinovirus", "score": 0.45},
                {"name": "Norovirus", "score": 0.31},
                {"name": "Rotavirus", "score": 0.18},
            ],
            "topSequences": [
                {"source": "human_virus_class1", "classLabel": "Class 1", "maxAct": 8.42, "tokens": _make_tokens("ATGCGTACGATCGATCGTAGCTAGCTGATCGATCGATCGTAGCTAGCTGA", (12, 28))},
                {"source": "human_virus_class2", "classLabel": "Class 2", "maxAct": 7.91, "tokens": _make_tokens("GCTAGCTAGCGATCGATCGATCGTAGCTAGCTGATCGATCGATCGTAGCT", (10, 26))},
                {"source": "human_virus_class1", "classLabel": "Class 1", "maxAct": 6.55, "tokens": _make_tokens("TGATCGATCGTAGCTAGCTGATCGATCGATCGTAGCTAGCTGATCGATCG", (8, 24))},
            ],
        },
        {
            "id": 842,
            "label": "Bacterial 16S signature",
            "description": "Fires on conserved 16S ribosomal RNA regions. Strongest for gram-negative bacteria.",
            "layer": 20,
            "freqActive": 0.087,
            "maxAct": 12.1,
            "category": "Domain",
            "activationPattern": "Domain",
            "histogram": _make_histogram(),
            "topTaxa": [
                {"name": "E. coli", "score": 0.91},
                {"name": "Klebsiella", "score": 0.78},
                {"name": "Pseudomonas", "score": 0.65},
                {"name": "Salmonella", "score": 0.52},
                {"name": "Bacteroides", "score": 0.41},
            ],
            "topSequences": [
                {"source": "hmpd_source", "classLabel": "Bacteria", "maxAct": 12.1, "tokens": _make_tokens("AGAGTTTGATCCTGGCTCAGATTGAACGCTGGCGGCATGCCTAACACATG", (0, 35))},
                {"source": "hmpd_disease", "classLabel": "Bacteria", "maxAct": 10.8, "tokens": _make_tokens("CCTGGCTCAGATTGAACGCTGGCGGCATGCCTAACACATGCAAGTCGAAC", (0, 38))},
            ],
        },
        {
            "id": 2041,
            "label": "GC-rich region detector",
            "description": "Responds to high GC-content stretches. May correspond to thermophilic organisms.",
            "layer": 12,
            "freqActive": 0.142,
            "maxAct": 5.67,
            "category": "Whole",
            "activationPattern": "Whole",
            "histogram": _make_histogram(),
            "topTaxa": [
                {"name": "Thermus thermophilus", "score": 0.73},
                {"name": "Deinococcus radiodurans", "score": 0.61},
                {"name": "Streptomyces", "score": 0.54},
                {"name": "Mycobacterium", "score": 0.38},
            ],
            "topSequences": [
                {"source": "hmpd_source", "classLabel": None, "maxAct": 5.67, "tokens": _make_tokens("GCCGCGCCGCGCCGCGGCCGCGCCGCCGCGCCGCGCCGCGGCCGCGCCG", (0, 50))},
                {"source": "hvr_default", "classLabel": None, "maxAct": 4.89, "tokens": _make_tokens("CCGCGGCCGCGCCGCCGCGCCGCGCCGCGGCCGCGCCGCCGCGCCGCGCC", (0, 50))},
            ],
        },
        {
            "id": 3599,
            "label": "Phage integrase site",
            "description": "Activates at attachment sites (attP/attB) for temperate phage integration.",
            "layer": 24,
            "freqActive": 0.012,
            "maxAct": 15.3,
            "category": "Point",
            "activationPattern": "Point",
            "histogram": _make_histogram(),
            "topTaxa": [
                {"name": "Lambda phage", "score": 0.88},
                {"name": "P22 phage", "score": 0.72},
                {"name": "Mu phage", "score": 0.44},
            ],
            "topSequences": [
                {"source": "human_virus_class1", "classLabel": "Phage", "maxAct": 15.3, "tokens": _make_tokens("ATCGATCGATCGATCGCTTTGCATTAGCTGATCGATCGATCGATCGATCG", (18, 20))},
                {"source": "human_virus_class2", "classLabel": "Phage", "maxAct": 13.7, "tokens": _make_tokens("GATCGATCGATCGCTTTGCATTAGCTGATCGATCGATCGATCGATCGATC", (17, 19))},
            ],
        },
        {
            "id": 1456,
            "label": "Human microbiome core",
            "description": "Broadly activated across human gut microbiome samples. Low specificity, high frequency.",
            "layer": 16,
            "freqActive": 0.312,
            "maxAct": 3.21,
            "category": "Whole",
            "activationPattern": "Whole",
            "histogram": _make_histogram(),
            "topTaxa": [
                {"name": "Bacteroides fragilis", "score": 0.55},
                {"name": "Faecalibacterium", "score": 0.51},
                {"name": "Prevotella", "score": 0.48},
                {"name": "Ruminococcus", "score": 0.42},
                {"name": "Bifidobacterium", "score": 0.39},
            ],
            "topSequences": [
                {"source": "hmpd_source", "classLabel": "Gut", "maxAct": 3.21, "tokens": _make_tokens("ATGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC", (0, 50))},
                {"source": "hmpd_disease", "classLabel": "Gut", "maxAct": 2.98, "tokens": _make_tokens("GATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGA", (0, 50))},
            ],
        },
    ]


# ── Experiment 1: Organism-Specific Pathogen Detectors ───────────────────────

def generate_experiment1() -> dict:
    """
    Schema:
    {
        summary: {
            total_latents: int,
            alive_latents: int,
            pathogen_enriched: int,      # FDR < 0.01, OR > 1
            nonpathogen_enriched: int,
            high_f1_latents: int,         # max F1 > 0.7
        },
        volcano: [{                       # all 32k latents (sampled for dummy)
            latent_id: int,
            log2fc: float,
            neg_log10_pval: float,
            significant: bool,
            direction: "pathogen"|"nonpathogen"|"ns"
        }],
        top_detectors: [{                 # top ~15 organism-labeled latents
            latent_id: int,
            odds_ratio: float,
            fdr_pval: float,
            log2fc: float,
            max_f1: float,
            dominant_organism: str,
            hit_consistency: str,         # e.g. "9/10 Influenza A"
            proposed_label: str,
            top_sequences: [{
                sequence_id: str,
                blast_organism: str,
                percent_identity: float,
                evalue: str,
                gene_annotation: str
            }]
        }],
        enrichment_histogram: [{          # distribution of odds ratios
            bin_start: float,
            bin_end: float,
            count: int
        }]
    }
    """
    # Volcano plot: simulate 2000 points (subset of 32k for frontend perf)
    volcano = []
    for i in range(2000):
        log2fc = random.gauss(0, 1.2)
        pval = 10 ** (-abs(log2fc) * random.uniform(0.5, 4))
        neg_log10 = -math.log10(max(pval, 1e-20))
        sig = neg_log10 > 2 and abs(log2fc) > 1
        direction = "pathogen" if sig and log2fc > 0 else ("nonpathogen" if sig and log2fc < 0 else "ns")
        volcano.append({
            "latent_id": i * 16,
            "log2fc": round(log2fc, 3),
            "neg_log10_pval": round(neg_log10, 3),
            "significant": sig,
            "direction": direction,
        })

    top_detectors = [
        {"latent_id": 7241, "odds_ratio": 4.2, "fdr_pval": 1.2e-15, "log2fc": 2.07, "max_f1": 0.84,
         "dominant_organism": "Influenza A virus", "hit_consistency": "9/10 Influenza A",
         "proposed_label": "Influenza A polymerase detector",
         "top_sequences": [
             {"sequence_id": "seq_14023", "blast_organism": "Influenza A virus (H1N1)", "percent_identity": 96.2, "evalue": "2.1e-45", "gene_annotation": "segment 2, polymerase PB1"},
             {"sequence_id": "seq_14089", "blast_organism": "Influenza A virus (H3N2)", "percent_identity": 94.8, "evalue": "8.3e-42", "gene_annotation": "segment 2, polymerase PB1"},
             {"sequence_id": "seq_14201", "blast_organism": "Influenza A virus (H1N1)", "percent_identity": 95.1, "evalue": "1.5e-43", "gene_annotation": "segment 2, polymerase PB1"},
         ]},
        {"latent_id": 12803, "odds_ratio": 3.8, "fdr_pval": 3.4e-12, "log2fc": 1.93, "max_f1": 0.79,
         "dominant_organism": "SARS-CoV-2", "hit_consistency": "8/10 SARS-CoV-2",
         "proposed_label": "Coronavirus spike detector",
         "top_sequences": [
             {"sequence_id": "seq_8412", "blast_organism": "SARS-CoV-2", "percent_identity": 99.1, "evalue": "0.0", "gene_annotation": "spike glycoprotein S"},
             {"sequence_id": "seq_8501", "blast_organism": "SARS-CoV-2", "percent_identity": 98.7, "evalue": "0.0", "gene_annotation": "spike glycoprotein S"},
         ]},
        {"latent_id": 5190, "odds_ratio": 3.5, "fdr_pval": 7.8e-11, "log2fc": 1.81, "max_f1": 0.76,
         "dominant_organism": "Norovirus GII", "hit_consistency": "7/10 Norovirus",
         "proposed_label": "Norovirus capsid detector",
         "top_sequences": [
             {"sequence_id": "seq_3201", "blast_organism": "Norovirus GII.4", "percent_identity": 93.4, "evalue": "4.2e-38", "gene_annotation": "VP1 capsid protein"},
         ]},
        {"latent_id": 22104, "odds_ratio": 3.1, "fdr_pval": 2.1e-9, "log2fc": 1.63, "max_f1": 0.73,
         "dominant_organism": "Rotavirus A", "hit_consistency": "8/10 Rotavirus",
         "proposed_label": "Rotavirus VP6 detector",
         "top_sequences": [
             {"sequence_id": "seq_1102", "blast_organism": "Rotavirus A", "percent_identity": 91.8, "evalue": "6.1e-35", "gene_annotation": "VP6 inner capsid"},
         ]},
        {"latent_id": 9871, "odds_ratio": 2.9, "fdr_pval": 5.5e-8, "log2fc": 1.54, "max_f1": 0.71,
         "dominant_organism": "Human adenovirus", "hit_consistency": "7/10 Adenovirus",
         "proposed_label": "Adenovirus hexon detector",
         "top_sequences": [
             {"sequence_id": "seq_6723", "blast_organism": "Human adenovirus C", "percent_identity": 95.6, "evalue": "1.3e-41", "gene_annotation": "hexon protein"},
         ]},
        {"latent_id": 18432, "odds_ratio": 2.7, "fdr_pval": 1.2e-7, "log2fc": 1.43, "max_f1": 0.68,
         "dominant_organism": "Hepatitis B virus", "hit_consistency": "6/10 HBV",
         "proposed_label": "HBV surface antigen region",
         "top_sequences": [
             {"sequence_id": "seq_11234", "blast_organism": "Hepatitis B virus", "percent_identity": 97.3, "evalue": "0.0", "gene_annotation": "surface antigen HBsAg"},
         ]},
        {"latent_id": 3047, "odds_ratio": 2.5, "fdr_pval": 8.9e-7, "log2fc": 1.32, "max_f1": 0.65,
         "dominant_organism": "HIV-1", "hit_consistency": "6/10 HIV-1",
         "proposed_label": "HIV-1 integrase region",
         "top_sequences": [
             {"sequence_id": "seq_5501", "blast_organism": "HIV-1", "percent_identity": 92.1, "evalue": "3.7e-36", "gene_annotation": "pol polyprotein, integrase"},
         ]},
        {"latent_id": 27615, "odds_ratio": 2.3, "fdr_pval": 4.5e-6, "log2fc": 1.20, "max_f1": 0.62,
         "dominant_organism": "Rhinovirus", "hit_consistency": "5/10 Rhinovirus",
         "proposed_label": "Rhinovirus generic",
         "top_sequences": []},
        {"latent_id": 14982, "odds_ratio": 2.1, "fdr_pval": 2.3e-5, "log2fc": 1.07, "max_f1": 0.59,
         "dominant_organism": "Enterovirus", "hit_consistency": "5/10 Enterovirus",
         "proposed_label": "Enterovirus polyprotein",
         "top_sequences": []},
        {"latent_id": 31200, "odds_ratio": 1.9, "fdr_pval": 8.7e-5, "log2fc": 0.93, "max_f1": 0.55,
         "dominant_organism": "Mixed viral", "hit_consistency": "3/10 mixed",
         "proposed_label": "Generic viral RdRp motif",
         "top_sequences": []},
    ]

    enrichment_hist = []
    for i in range(20):
        lo = round(i * 0.5, 1)
        hi = round((i + 1) * 0.5, 1)
        count = int(8000 * math.exp(-((lo - 1.0) ** 2) / 0.8)) + random.randint(0, 200)
        enrichment_hist.append({"bin_start": lo, "bin_end": hi, "count": count})

    pathogen_enriched = sum(1 for v in volcano if v["direction"] == "pathogen")
    nonpathogen_enriched = sum(1 for v in volcano if v["direction"] == "nonpathogen")

    return {
        "summary": {
            "total_latents": 32768,
            "alive_latents": 31965,
            "pathogen_enriched": pathogen_enriched,
            "nonpathogen_enriched": nonpathogen_enriched,
            "high_f1_latents": 47,
        },
        "volcano": volcano,
        "top_detectors": top_detectors,
        "enrichment_histogram": enrichment_hist,
    }


# ── Experiment 2: Linear Probe ───────────────────────────────────────────────

def generate_experiment2() -> dict:
    """
    Schema:
    {
        summary: {accuracy, mcc, auroc, best_C, n_train, n_test, n_features},
        roc_curve: [{fpr: float, tpr: float}],
        coefficient_distribution: [{bin_center: float, count: int}],
        top_latents: [{
            latent_id, coefficient, direction, freq_pathogen, freq_nonpathogen,
            mean_act_pathogen, mean_act_nonpathogen, enrichment
        }]
    }
    """
    roc = []
    for i in range(101):
        fpr = i / 100
        tpr = min(1.0, fpr ** 0.3)  # decent-looking curve
        roc.append({"fpr": round(fpr, 3), "tpr": round(tpr, 3)})

    coef_dist = []
    for i in range(40):
        center = round(-2.0 + i * 0.1, 2)
        count = int(4000 * math.exp(-(center ** 2) / 0.3)) + random.randint(0, 50)
        coef_dist.append({"bin_center": center, "count": count})

    top_latents = [
        {"latent_id": 7241, "coefficient": 1.82, "direction": "pathogen", "freq_pathogen": 0.41, "freq_nonpathogen": 0.08, "mean_act_pathogen": 2.34, "mean_act_nonpathogen": 0.12, "enrichment": 5.13},
        {"latent_id": 12803, "coefficient": 1.65, "direction": "pathogen", "freq_pathogen": 0.38, "freq_nonpathogen": 0.11, "mean_act_pathogen": 1.98, "mean_act_nonpathogen": 0.21, "enrichment": 3.45},
        {"latent_id": 5190, "coefficient": 1.43, "direction": "pathogen", "freq_pathogen": 0.29, "freq_nonpathogen": 0.07, "mean_act_pathogen": 1.76, "mean_act_nonpathogen": 0.09, "enrichment": 4.14},
        {"latent_id": 22104, "coefficient": 1.31, "direction": "pathogen", "freq_pathogen": 0.25, "freq_nonpathogen": 0.06, "mean_act_pathogen": 1.54, "mean_act_nonpathogen": 0.08, "enrichment": 4.17},
        {"latent_id": 9871, "coefficient": 1.18, "direction": "pathogen", "freq_pathogen": 0.22, "freq_nonpathogen": 0.09, "mean_act_pathogen": 1.33, "mean_act_nonpathogen": 0.14, "enrichment": 2.44},
        {"latent_id": 28401, "coefficient": -1.71, "direction": "nonpathogen", "freq_pathogen": 0.05, "freq_nonpathogen": 0.37, "mean_act_pathogen": 0.07, "mean_act_nonpathogen": 2.11, "enrichment": 0.14},
        {"latent_id": 15632, "coefficient": -1.54, "direction": "nonpathogen", "freq_pathogen": 0.08, "freq_nonpathogen": 0.34, "mean_act_pathogen": 0.11, "mean_act_nonpathogen": 1.87, "enrichment": 0.24},
        {"latent_id": 4087, "coefficient": -1.39, "direction": "nonpathogen", "freq_pathogen": 0.06, "freq_nonpathogen": 0.31, "mean_act_pathogen": 0.09, "mean_act_nonpathogen": 1.65, "enrichment": 0.19},
        {"latent_id": 19200, "coefficient": -1.25, "direction": "nonpathogen", "freq_pathogen": 0.09, "freq_nonpathogen": 0.28, "mean_act_pathogen": 0.13, "mean_act_nonpathogen": 1.42, "enrichment": 0.32},
        {"latent_id": 8845, "coefficient": -1.12, "direction": "nonpathogen", "freq_pathogen": 0.07, "freq_nonpathogen": 0.26, "mean_act_pathogen": 0.10, "mean_act_nonpathogen": 1.31, "enrichment": 0.27},
    ]

    return {
        "summary": {
            "accuracy": 0.8735,
            "mcc": 0.747,
            "auroc": 0.934,
            "best_C": 0.1,
            "n_train": 16000,
            "n_test": 4000,
            "n_features": 32768,
        },
        "roc_curve": roc,
        "coefficient_distribution": coef_dist,
        "top_latents": top_latents,
    }


# ── Experiment 3: SAE Health Check ───────────────────────────────────────────

def generate_experiment3() -> dict:
    """
    Schema:
    {
        summary: {
            total_latents, dead_count, alive_count, dead_pct,
            sparsity_pct, mean_active_per_seq, median_active_per_seq,
            mean_activation_count, median_activation_count
        },
        sequences_per_latent: [{bin_start, bin_end, count}],
        max_activation_dist: [{bin_start, bin_end, count}],
        active_features_per_seq: [{bin_center, count}],
        comparison: {
            interprot: {d_model, expansion, k, total_latents, dead_pct},
            ours: {d_model, expansion, k, total_latents, dead_pct}
        }
    }
    """
    # Sequences per latent — long tail
    seq_per_latent = []
    for i in range(20):
        lo = i * 1000
        hi = (i + 1) * 1000
        count = int(15000 * math.exp(-i * 0.35)) + random.randint(0, 500)
        seq_per_latent.append({"bin_start": lo, "bin_end": hi, "count": count})

    # Max activation distribution
    max_act_dist = []
    for i in range(25):
        lo = round(i * 2.0, 1)
        hi = round((i + 1) * 2.0, 1)
        count = int(5000 * math.exp(-((lo - 6) ** 2) / 20)) + random.randint(0, 200)
        max_act_dist.append({"bin_start": lo, "bin_end": hi, "count": count})

    # Active features per sequence — peak near k=64
    active_per_seq = []
    for i in range(20):
        center = 50 + i * 2
        count = int(3000 * math.exp(-((center - 64) ** 2) / 30)) + random.randint(0, 50)
        active_per_seq.append({"bin_center": center, "count": count})

    return {
        "summary": {
            "total_latents": 32768,
            "dead_count": 803,
            "alive_count": 31965,
            "dead_pct": 2.45,
            "sparsity_pct": 2.72,
            "mean_active_per_seq": 63.8,
            "median_active_per_seq": 64,
            "mean_activation_count": 389.2,
            "median_activation_count": 142,
        },
        "sequences_per_latent": seq_per_latent,
        "max_activation_dist": max_act_dist,
        "active_features_per_seq": active_per_seq,
        "comparison": {
            "interprot": {"d_model": 1280, "expansion": "2-4x", "k": 64, "total_latents": "4096-8192", "dead_pct": "varies"},
            "ours": {"d_model": 4096, "expansion": "8x", "k": 64, "total_latents": 32768, "dead_pct": 2.45},
        },
    }


# ── Experiment 4: Sequence UMAP ──────────────────────────────────────────────

def generate_experiment4() -> dict:
    """
    Schema:
    {
        points: [{x, y, label, sequence_id}],  # label: 0=non-pathogen, 1=pathogen
        pca_variance: [{component, explained_variance}],
        summary: {n_sequences, n_pathogen, n_nonpathogen, pca_dims, variance_explained_50}
    }
    """
    points = []
    for i in range(2000):
        label = 1 if i < 1000 else 0
        if label == 1:
            x = random.gauss(3.0, 1.5) + random.choice([0, 4]) * random.random()
            y = random.gauss(2.0, 1.2)
        else:
            x = random.gauss(-2.0, 1.8)
            y = random.gauss(-1.0, 1.5)
        points.append({
            "x": round(x, 3),
            "y": round(y, 3),
            "label": label,
            "sequence_id": f"seq_{i}",
        })

    pca_var = []
    remaining = 1.0
    for i in range(50):
        v = remaining * random.uniform(0.05, 0.25)
        remaining -= v
        pca_var.append({"component": i + 1, "explained_variance": round(v, 4)})

    return {
        "points": points,
        "pca_variance": pca_var,
        "summary": {
            "n_sequences": 20000,
            "n_pathogen": 10000,
            "n_nonpathogen": 10000,
            "pca_dims": 50,
            "variance_explained_50": round(1.0 - remaining, 3),
        },
    }


# ── Experiment 5: Feature Clustering ─────────────────────────────────────────

def generate_experiment5() -> dict:
    """
    Schema:
    {
        points: [{x, y, cluster_id, latent_id, enrichment, activation_count}],
        cluster_summary: [{
            cluster_id, size, mean_enrichment, mean_activation_count, label
        }],
        summary: {n_latents, n_clusters, noise_count}
    }
    """
    cluster_centers = [
        (2.0, 3.0, "Pathogen module"),
        (-3.0, 1.0, "Non-pathogen module"),
        (0.0, -2.5, "Broad features"),
        (4.0, -1.0, "Rare features"),
        (-1.0, 4.0, "16S ribosomal"),
        (-4.0, -3.0, "GC-content"),
    ]

    points = []
    cluster_sizes = [0] * len(cluster_centers)
    for i in range(3000):
        if random.random() < 0.08:
            # Noise
            x = random.uniform(-8, 8)
            y = random.uniform(-8, 8)
            cid = -1
            enrichment = round(random.uniform(0.5, 1.5), 2)
        else:
            ci = random.randint(0, len(cluster_centers) - 1)
            cx, cy, _ = cluster_centers[ci]
            x = random.gauss(cx, 0.8)
            y = random.gauss(cy, 0.8)
            cid = ci
            cluster_sizes[ci] += 1
            # Pathogen module has high enrichment
            if ci == 0:
                enrichment = round(random.uniform(2.0, 5.0), 2)
            elif ci == 1:
                enrichment = round(random.uniform(0.1, 0.5), 2)
            else:
                enrichment = round(random.uniform(0.7, 1.5), 2)

        points.append({
            "x": round(x, 3),
            "y": round(y, 3),
            "cluster_id": cid,
            "latent_id": i * 10,
            "enrichment": enrichment,
            "activation_count": random.randint(10, 5000),
        })

    cluster_summary = []
    for ci, (cx, cy, label) in enumerate(cluster_centers):
        members = [p for p in points if p["cluster_id"] == ci]
        if members:
            cluster_summary.append({
                "cluster_id": ci,
                "size": len(members),
                "mean_enrichment": round(sum(p["enrichment"] for p in members) / len(members), 3),
                "mean_activation_count": round(sum(p["activation_count"] for p in members) / len(members), 1),
                "label": label,
            })

    noise_count = sum(1 for p in points if p["cluster_id"] == -1)

    return {
        "points": points,
        "cluster_summary": cluster_summary,
        "summary": {
            "n_latents": 32768,
            "n_clusters": len(cluster_centers),
            "noise_count": noise_count,
        },
    }
