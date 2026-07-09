# Source Attribution of Human- and LLM-Generated Text

Multi-class source attribution on the RAID dataset (12 classes: human + 11 LLMs).
`src/` holds the stable, reusable logic; `notebooks/` drives exploration and produces
report figures. See the group proposal for the full research questions (RQ1–RQ4).

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
# put the RAID dump where config.DATA_RAW points, or edit data.load_raw()
```

## The one rule

Split **before** you extract features. Every transformer (TF-IDF vocab, scalers)
is fit on **train only**. Two artifacts get frozen and never rebuilt:
`data/clean.parquet` and `artifacts/split_indices.json`. If either changes, all
experiment results stop being comparable.

## Build order (single-person path)

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
