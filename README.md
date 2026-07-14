# Source Attribution of Human- and LLM-Generated Text

Multi-class source attribution on the RAID dataset (12 classes: human + 11 LLMs).
`src/` holds the stable, reusable logic; `notebooks/` runs the experiments and
produces every report figure/table; `FINAL_REPORT_OUTLINE.md` is the section-by-
section writing plan for the final report. Three research questions after
merging RQ2/RQ3 per supervisor feedback on the detailed proposal (RQ3's
confusion/lineage question was a natural output of investigating RQ2, not a
separate one) — see `FINAL_REPORT_OUTLINE.md` for the exact wording, and the
group's detailed proposal PDF for the original methodology this code
implements.

## Status: all experiments done, writing the final report

Every notebook `00`–`09` has run. Headline (held-out **test**-set) numbers, from
`08_test_evaluation`:

| | Binary (human vs AI) | 12-way (attribution) | Gap |
|---|---|---|---|
| Macro-F1 (test) | 0.901 | 0.772 | 0.128 |

Start with **`FINAL_REPORT_OUTLINE.md`** — it maps every report section to the
specific notebook, figure, and number to pull from. Every notebook also ends with
its own **Conclusion** markdown cell; read those first if you want the findings
without re-running anything.

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
experiment results stop being comparable — delete both and rerun `00` from
scratch if you ever need to regenerate them (e.g. after a preprocessing change).

`make_splits()` groups by `source_id` (no prompt leakage) and stratifies on
**(model, domain) jointly**, not model alone. Every `source_id` group already
contains exactly one row per class, so class balance across folds is guaranteed
by construction regardless of split assignment — domain is the thing that
actually varies group-to-group, and RQ3's domain-stratified analysis (3.3.7)
needs each split to be domain-balanced too. Verified: every domain lands within
~0.1pp of the same share in train/val/test.

**Test is held out completely** until `08_test_evaluation` — `03`–`07` compare
conditions on `splits['val']` only (correct for model selection), and test is
touched exactly once, for the numbers actually reported as final.

## Build order

| # | Notebook | Proposal | Produces / answers |
|---|----------|----------|--------------------|
| 0 | `00_data_and_split` | 3.1, 3.3.1 | frozen clean df + grouped split indices |
| 1 | `01_baseline_minimal` | — | minimal TF-IDF→LR loop; lights up the chain |
| 2 | `02_features` | 3.2 | stylometric / SBERT / Biber implementations (cached) + sanity check |
| 3 | `03_ablation` | 3.3.3 | Exp 1–6 metrics table (val) → **RQ1** |
| 4 | `04_detection_baseline` | 3.3.4 | binary vs 12-way gap (val) → **RQ1** |
| 5 | `05_error_analysis` | 3.3.5 | confusion + family clustering → **RQ2** |
| 6 | `06_embedding` | 3.3.6 | centroid similarity + domain-human distances → **RQ2** |
| 7 | `07_domain` | 3.3.7 | per-domain re-runs → **RQ3** |
| 8 | `08_test_evaluation` | 3.3.1 | held-out test evaluation → **RQ1 headline numbers** |
| 9 | `09_interpretability` | 3.2, 3.3.2 | logreg coefficients + class-level feature means → **RQ1/RQ2 mechanism** |

Work notebooks 0→1→2 in order (serial dependency: everything downstream reads
`00`'s cached clean df, and `02` is what first computes and caches the SBERT
embeddings / Biber POS tags that `03`+ all reuse). After that, `03`–`09` are
each self-contained — every one rebuilds its own feature blocks from cache
rather than depending on another notebook's kernel state, so `Restart & Run
All` on any of them works standalone.

**Runtime, so you can plan around it**: `03` is the expensive one (~2–2.5h — 13
classifier fits, several on 50,000+ dims). `04`/`05`/`08` each cost ~40–45 min
(one or two Exp6-scale fits). `06`/`07`/`09` are cheaper (~15–45 min). All of
them can run unattended once started — none hang (see below for why they used
to).

## `src/` modules

- `config.py` — classes, domains, model families, paths, seed, feature params
- `utils.py` — `set_seed`, `cache_or_compute` (the caching backbone), `save_fig`
- `data.py` — RAID loader, preprocessing (3.1), grouped split (3.3.1)
- `features.py` — TF-IDF / stylometric / Biber / SBERT + `assemble` for Exp 1–6
- `modeling.py` — `train_and_evaluate` (the workhorse), classifier factory
- `analysis.py` — error analysis, embedding similarity, domain-stratified helpers
- `viz.py` — shared matplotlib styling (palette, `new_fig`) so every notebook's
  figures read as one report

## Two fixed bugs worth knowing about before touching `features.py`/`modeling.py`

1. **Solver choice.** `LogisticRegression` uses `solver="saga"` (not sklearn's
   default `lbfgs`) and `LinearSVC` uses `dual="auto"`. lbfgs / liblinear's old
   default effectively never converge on the TF-IDF-scale (50,000+ dim) sparse
   conditions — one fit ran 10+ hours before this fix and still hadn't finished.
2. **Block-scale harmonization.** `features.assemble()` scales each feature
   block (tfidf, stylometric, biber, sbert, length) by its own RMS, computed on
   train rows, before concatenating. Without this, TF-IDF's tiny row-normalized
   values (~0.05–0.3) and the standardized dense blocks (~unit variance) sit on
   wildly different scales, which destabilizes the optimizer — measured at
   ~0.3 Macro-F1 lost on the combined conditions, on top of far slower/
   non-convergent fits. **Do not** "fix" this with a per-feature
   `StandardScaler` on the already-assembled matrix — that was tried first and
   made it *worse*: TF-IDF has 50,000 mostly-rare columns, so per-feature
   scaling divides by a near-zero column std and inflates rare terms ~100×,
   which is what caused the 10-hour hang in the first place.

## Other gotchas baked into the code

1. **sparse + dense.** TF-IDF is sparse, the other three blocks are dense.
   `features.assemble` uses `scipy.sparse.hstack` when TF-IDF is present.
2. **MultinomialNB only on pure TF-IDF (Exp1).** NB needs non-negative features
   everywhere; scaled dense blocks are mean-centered, so any experiment mixing
   them in would violate that even where TF-IDF is also present (Exp4/Exp6
   don't get NB).
3. **Never quote val-set numbers as "final."** `03`–`07` compare conditions on
   `splits['val']` — correct for model selection, but `08` is what actually
   evaluates on `splits['test']`, which nothing else ever touches. Report `08`'s
   numbers, not `03`/`04`'s, as the headline result (they're close — within
   0.005 Macro-F1 — which is itself worth a sentence in the report as evidence
   val-based selection generalized).

## Working notes

- Every notebook: `set_seed()` at the top, save tables/figures to `artifacts/`
  or `figures/` at the bottom, and ends with a **Conclusion** markdown cell.
- `%autoreload 2` is set so edits to `src/` take effect **between** cell runs —
  not mid-cell. If a long-running cell is already executing when you edit
  `src/`, it won't pick up the change until that cell finishes and you rerun it.
- `figures/*.png` and `artifacts/*.csv` are tracked in git (not ignored) — they
  are the actual report deliverables, not throwaway build output. `artifacts/
  *.pkl`/`*.json` and `data/*.parquet`/`*.csv` stay ignored (regenerable from
  cache / too large for the repo).
- Before submitting, run each notebook **Restart & Run All** to confirm no
  hidden execution-order dependencies.
