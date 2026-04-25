# Log

Reverse-chronological. Each session appends what changed, what's unfinished, what to pick up next.

### 2026-04-25
- Train/test split: train set is class 1 of `human_virus_infecting` (`data/human_virus_class1.jsonl`, 20k seqs), test set is class 2 (`data/human_virus_class2.jsonl`, 20k seqs)
- Bridget ran inference on the class 1 file (the label-stripped version Ciaran and Astrid prepped) using the cloud prod config
- Labels (`source`, `class`) exist in `data/raw_sources/human_virus_infecting.csv` and `data/curated_sequences/forward_pass_unique_sequences.jsonl` but were not included in the per-class JSONL files
