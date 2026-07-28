"""Central configuration for the Argo RAG pipeline.

Every value can be overridden with an environment variable so the same code
runs on a laptop, a lab workstation or CI without editing source.
Paths are resolved relative to this file, so no absolute Windows paths leak
into the repository.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _path(env_var: str, default: str) -> Path:
    raw = os.getenv(env_var)
    return Path(raw).expanduser() if raw else BASE_DIR / default


def _int(env_var: str, default: int) -> int:
    return int(os.getenv(env_var, default))


def _float(env_var: str, default: float) -> float:
    return float(os.getenv(env_var, default))


# ── Data / artefacts ────────────────────────────────────────────────
CSV_PATH = _path("ARGO_CSV_PATH", "argo_preprocessed_with_dates.csv")
VECTORSTORE_PATH = _path("ARGO_VECTORSTORE_PATH", "weather_faiss_vectorstore_main")
# Tokenised BM25 corpus, cached next to the vectorstore so start-up does not
# have to re-tokenise every document on each run.
BM25_CACHE_PATH = _path("ARGO_BM25_CACHE", "weather_faiss_vectorstore_main/bm25_cache.pkl")

# ── Embeddings ──────────────────────────────────────────────────────
EMBED_MODEL = os.getenv("ARGO_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DEVICE = os.getenv("ARGO_EMBED_DEVICE", "cpu")
# The index was built from L2-normalised vectors; keep this True or retrieval
# scores stop being comparable with the stored index.
NORMALIZE_EMBEDDINGS = True

# ── Index build ─────────────────────────────────────────────────────
CSV_CHUNK_SIZE = _int("ARGO_CSV_CHUNK_SIZE", 10_000)
EMBED_BATCH_SIZE = _int("ARGO_EMBED_BATCH_SIZE", 32)

# "flat" gives exact cosine search; at this corpus size (~150k x 384) that is
# tens of milliseconds, and it removes the nlist/nprobe tuning surface
# entirely. Switch to "ivf" past roughly a million rows.
INDEX_TYPE = os.getenv("ARGO_INDEX_TYPE", "flat").lower()
# Rule of thumb from the FAISS wiki: nlist ~ sqrt(n). The original build used
# min(100, n//10), which caps at 100 cells -- with ~150k vectors each cell
# holds ~1500 of them, so nprobe=10 scanned about a tenth of the corpus.
IVF_NLIST_CAP = _int("ARGO_IVF_NLIST_CAP", 4096)
NPROBE = _int("ARGO_NPROBE", 10)

# ── Retrieval ───────────────────────────────────────────────────────
TOP_K = _int("ARGO_TOP_K", 6)
FETCH_K = _int("ARGO_FETCH_K", 20)
MMR_LAMBDA = _float("ARGO_MMR_LAMBDA", 0.7)
# Weights for reciprocal-rank fusion of the dense and lexical retrievers.
DENSE_WEIGHT = _float("ARGO_DENSE_WEIGHT", 0.5)
LEXICAL_WEIGHT = _float("ARGO_LEXICAL_WEIGHT", 0.5)
RRF_K = _int("ARGO_RRF_K", 60)

# ── Generation ──────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("ARGO_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("ARGO_OLLAMA_MODEL", "qwen3:latest")
LLM_NUM_PREDICT = _int("ARGO_NUM_PREDICT", 2048)
LLM_TEMPERATURE = _float("ARGO_TEMPERATURE", 0.1)
LLM_TOP_P = _float("ARGO_TOP_P", 0.75)
LLM_TIMEOUT = _int("ARGO_LLM_TIMEOUT", 300)
# Ollama defaults to a 2048-4096 token context depending on the model file.
# Six stuffed records plus this prompt can exceed that, and the overflow is
# dropped silently -- the model then answers from a truncated context.
LLM_NUM_CTX = _int("ARGO_NUM_CTX", 8192)

# ── Corpus schema ───────────────────────────────────────────────────
# Actual headers of argo_preprocessed_with_dates.csv, lower-cased. These drive
# both document rendering and the field list shown to the model.
PRIORITY_COLS = [
    "datetime",
    "latitude",
    "longitude",
    "pres",
    "temp",
    "psal",
    "platform_number",
    "cycle_number",
]
SKIP_COLS = {"row_id", "index", "unnamed: 0", "juld", "n_prof", "n_levels"}

# Human-readable labels + units, used when rendering a row and when telling the
# LLM what each field means. Keeping one source of truth stops the prompt from
# describing fields the corpus does not actually contain.
FIELD_LABELS = {
    "datetime": ("Date", "YYYY-MM-DD"),
    "latitude": ("Latitude", "decimal degrees, negative = South"),
    "longitude": ("Longitude", "decimal degrees, negative = West"),
    "pres": ("Pressure", "dbar, approx. equal to depth in metres"),
    "temp": ("Temperature", "deg C"),
    "psal": ("Salinity", "PSU"),
    "platform_number": ("Platform Number", "float identifier"),
    "cycle_number": ("Cycle Number", "profile number for that float"),
}
