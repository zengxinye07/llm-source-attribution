"""Shared configuration: constants every other module imports.

Single source of truth for class names, domains, paths, and reproducibility
settings. Do NOT redefine these anywhere else.
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# Classes (proposal Section 3.1) -- 12 classes total
# One aggregated human class + 11 individual LLM classes.
# --------------------------------------------------------------------------- #
HUMAN_CLASS = "human"

LLM_CLASSES = [
    "chatgpt",
    "gpt4",
    "gpt3",
    "gpt2",
    "llama-chat",
    "mistral",
    "mistral-chat",
    "mpt",
    "mpt-chat",
    "cohere",
    "cohere-chat",
]

CLASSES = [HUMAN_CLASS] + LLM_CLASSES  # length 12
assert len(CLASSES) == 12

# Expected balanced counts (proposal Section 3.1) -- use for assertions.
N_PER_CLASS = 13_371
N_TOTAL = 160_452
assert N_PER_CLASS * len(CLASSES) == N_TOTAL

# --------------------------------------------------------------------------- #
# Domains (proposal Section 3.3.7) -- 8 RAID domains
# --------------------------------------------------------------------------- #
DOMAINS = [
    "abstracts",
    "books",
    "news",
    "poetry",
    "recipes",
    "reddit",
    "reviews",
    "wikipedia",
]

# --------------------------------------------------------------------------- #
# Model families (proposal RQ3) -- for within- vs cross-family confusion.
# Human is intentionally its own family.
# --------------------------------------------------------------------------- #
MODEL_FAMILIES = {
    "human": "human",
    "chatgpt": "openai",
    "gpt4": "openai",
    "gpt3": "openai",
    "gpt2": "openai",
    "llama-chat": "meta",
    "mistral": "mistral",
    "mistral-chat": "mistral",
    "mpt": "mosaic",
    "mpt-chat": "mosaic",
    "cohere": "cohere",
    "cohere-chat": "cohere",
}

# --------------------------------------------------------------------------- #
# Split (proposal Section 3.3.1) -- grouped by source_id, 70/15/15
# --------------------------------------------------------------------------- #
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
GROUP_COL = "source_id"
LABEL_COL = "model"   # RAID column holding the source-model label
DOMAIN_COL = "domain"

# --------------------------------------------------------------------------- #
# Feature params
# --------------------------------------------------------------------------- #
TFIDF_PARAMS = dict(ngram_range=(1, 2), min_df=5, max_features=50_000, sublinear_tf=True)
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
SPACY_MODEL_NAME = "en_core_web_sm"
MIN_TOKENS = 3  # drop texts with fewer than this many tokens (likely corrupted)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_RAW = DATA_DIR / "raid_raw.parquet"          # TODO: point at your RAID download
CLEAN_PARQUET = DATA_DIR / "clean.parquet"         # frozen output of preprocessing
ARTIFACTS = ROOT / "artifacts"                      # cached features / embeddings / splits
FIGURES = ROOT / "figures"                          # report-bound plots
SPLIT_INDEX_PATH = ARTIFACTS / "split_indices.json" # frozen train/val/test row indices

for _d in (DATA_DIR, ARTIFACTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)
