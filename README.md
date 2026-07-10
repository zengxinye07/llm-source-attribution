# Source Attribution of Human- and LLM-Generated Text

Multi-class source attribution on the RAID dataset (12 classes: human + 11 LLMs).
`src/` holds the stable, reusable logic; `notebooks/` drives exploration and produces
report figures. See the group proposal for the full research questions (RQ1–RQ4).

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
# put RAID's train_none.csv at data/train_none.csv (config.DATA_RAW).
# test_none.csv has no label columns (RAID's hidden-label leaderboard split) and
# extra_none.csv is unused -- train_none.csv is our only labeled source; we carve
# our own train/val/test out of it (see "The one rule" below).
```

`attack` is a RAID column marking whether a generation went through an adversarial
perturbation (typos, paraphrase, etc.) meant to evade detectors. We filter to
`attack == "none"` (proposal 3.1) so the classifier learns source-specific style,
not artifacts of an evasion attack.

## The one rule

Split **before** you extract features. Every transformer (TF-IDF vocab, scalers)
is fit on **train only**. Two artifacts get frozen and never rebuilt:
`data/clean.parquet` and `artifacts/split_indices.json`. If either changes, all
experiment results stop being comparable.

`make_splits()` groups by `source_id` (no prompt leakage) and stratifies on
**(model, domain) jointly**, not model alone. Every source_id group already
contains exactly one row per class, so class balance across folds is guaranteed
by construction regardless of split assignment -- domain is the thing that
actually varies group-to-group, and RQ4's domain-stratified analysis (3.3.7)
needs each split to be domain-balanced too. Verified: every domain lands within
~0.1pp of the same share in train/val/test.

## Build order 

| # | Notebook | Proposal | Produces / answers |
|---|----------|----------|--------------------|
| 0 | `00_data_and_split` | 3.1, 3.3.1 | frozen clean df + grouped split indices |
| 1 | `01_baseline_minimal` | — | minimal TF-IDF→LR loop; lights up the chain |
| 2 | `02_features` | 3.2 | stylometric → sbert → biber (cached) |
| 3 | `03_ablation` | 3.3.3 | Exp 1–6 metrics table → **RQ1** |
| 4 | `04_detection_baseline` | 3.3.4 | binary vs 12-way gap → **RQ1** |
| 5 | `05_error_analysis` | 3.3.5 | confusion + family clustering → **RQ3** |
| 6 | `06_embedding` | 3.3.6 | centroid similarity + domain distances → **RQ2/RQ3** |
| 7 | `07_domain` | 3.3.7 | per-domain re-runs → **RQ4** |

Work notebooks 0→1→2 in order (serial dependency). After that, 3–7 are largely
independent and read cached artifacts.

**If time runs short, cut `06_embedding` first.** It's a nice-to-have, not
load-bearing: the confusion-matrix analysis in `05_error_analysis` already
covers RQ2/RQ3 using all feature types, whereas the embedding centroid analysis
is SBERT-only. Supervisor feedback on the proposal flagged this explicitly.

## `src/` modules

- `config.py` — classes, domains, model families, paths, seed, feature params
- `utils.py` — `set_seed`, `cache_or_compute` (the caching backbone), `save_fig`
- `data.py` — RAID loader, preprocessing (3.1), grouped split (3.3.1)
- `features.py` — TF-IDF / stylometric / Biber / SBERT + `assemble` for Exp 1–6
- `modeling.py` — `train_and_evaluate` (the workhorse), classifier factory
- `analysis.py` — error analysis, embedding similarity, domain-stratified helpers

## Two gotchas baked into the code

1. **sparse + dense.** TF-IDF is sparse, the other three blocks are dense.
   `features.assemble` uses `scipy.sparse.hstack` when TF-IDF is present. Only
   dense blocks get scaled — never scale TF-IDF.
2. **MultinomialNB only on TF-IDF.** NB needs non-negative features, so it's
   restricted to the TF-IDF conditions (proposal 3.2).

## Working notes

- Every notebook: `set_seed()` at the top, save tables/figures to `artifacts/`
  or `figures/` at the bottom. Don't leave report outputs only in cell output.
- Before submitting, run each notebook **Restart & Run All** to confirm no hidden
  execution-order dependencies.
- `%autoreload 2` is set so edits to `src/` take effect without a kernel restart.

## Proposal-doc TODOs (supervisor feedback, not code)

These are fixes to the *written proposal* (PDF), not this codebase — left here so
they don't get lost before the next revision:

- **Tighten RQ1–RQ3.** RQ3 (shared training lineage / alignment / architecture)
  is really an output of investigating RQ2, not a separate question — consider
  folding it in or rephrasing to remove the overlap.
- **§2 background, statistical/metric-based detectors:** claim they have "strong
  interpretability" without saying why. Add a sentence justifying it (e.g.
  perplexity/log-prob thresholds map directly to an inspectable score) or drop
  the claim.
- **§2 background, supervised classifier paragraph:** the sentence on "pre-trained
  language models have poor generalisation ability" doesn't follow from the
  preceding sentence about supervised classifiers — pretrained LMs aren't the
  same thing as trained supervised classifiers. Rewrite the transition.
- **§3.2/3.3.4:** add a line clarifying what the `attack` column actually
  represents (see the Setup section above for the working explanation) — this
  is now documented in code but should also appear in the proposal text.
