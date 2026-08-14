# Argo-RAG

Retrieval-Augmented Generation over Argo oceanographic float profiles.

Ask questions in plain English about temperature, salinity and pressure
measurements; the pipeline retrieves the relevant profile records and has a
local LLM answer strictly from them. Everything runs offline.

## How it works

```
CSV row ──► labelled Document ──► MiniLM embedding ──► FAISS index
                                                          │
query ──► date normalisation ──┬──► FAISS (dense, MMR) ───┤
                               └──► BM25 (lexical) ───────┴──► RRF fusion
                                                                  │
                                                       prompt ──► Ollama ──► answer + sources
```

Retrieval is hybrid: a dense vector search catches paraphrases, BM25 catches
exact tokens like float ids and ISO dates, and the two ranked lists are merged
with weighted reciprocal-rank fusion. Queries are rewritten first, so "January
3rd, 2023" becomes "2023-01-03" and matches the indexed text.

## Setup

```bash
pip install -r requirements.txt
ollama serve                 # in another terminal
ollama pull qwen3

python RAG_main.py --check   # verify prerequisites
```

`--check` reports exactly what is missing and how to fix it.

## Usage

```bash
python RAG_main.py "What was the temperature on January 3rd, 2023 near 45N 30W?"
python RAG_main.py --quiet "salinity at platform 1902029"
python RAG_main.py --no-bm25 -k 10 "deep profiles in the Southern Ocean"

streamlit run app.py         # dashboard
```

Building the index from a CSV:

```bash
python embed_gen.py                  # full build
python embed_gen.py --limit 5000     # quick smoke build
python embed_gen.py --dry-run        # render documents, embed nothing
```

## Configuration

Everything lives in `config.py` and every value has an environment override, so
no paths are hard-coded:

```bash
ARGO_TOP_K=10 ARGO_OLLAMA_MODEL=llama3.2 python RAG_main.py "..."
```

Common knobs: `ARGO_TOP_K`, `ARGO_DENSE_WEIGHT` / `ARGO_LEXICAL_WEIGHT`,
`ARGO_NUM_CTX`, `ARGO_INDEX_TYPE` (`flat` or `ivf`), `ARGO_VECTORSTORE_PATH`.

## Files

| File | Purpose |
|---|---|
| `config.py` | All settings, env-overridable |
| `retrieval.py` | Date normalisation, RRF fusion, index/BM25 loading |
| `RAG_main.py` | LCEL chain, prompt, CLI, `main()` used by the dashboard |
| `embed_gen.py` | Builds the FAISS vectorstore from the CSV |
| `app.py` | Streamlit dashboard |
| `tests/` | Unit tests for the retrieval logic (`python -m pytest tests/ -q`) |
| `Doc_setup.py` | Superseded flan-t5 prototype, kept for reference |

## Known limitations

**Aggregate questions are not computed.** "What was the average temperature?"
cannot be answered by top-k retrieval — six retrieved rows out of ~150,000 are
not a representative sample, and averaging them would produce a confident wrong
number. The pipeline detects these questions and instructs the model to say so
rather than confabulate. Answering them properly needs a SQL path over the full
dataset (planned).

**No metadata pre-filtering yet.** Date and bounding-box filters need FAISS
`IDSelector` support; LangChain's `filter=` argument only post-filters the
`fetch_k` candidates, so a selective filter usually returns nothing at all.

**The BM25 cache is a pickle**, rebuilt automatically if it fails to load. A
SQLite FTS5 index would be more robust and would enable filtering in the same
query.

## Notes on the index

Embeddings are L2-normalised, so inner product is cosine similarity. New builds
use `IndexFlatIP` (exact search — at this corpus size the IVF approximation
bought nothing while adding two tuning knobs). An existing L2 index still works
and returns the same ranking: for unit vectors `||q−d||² = 2 − 2(q·d)`, so L2
and inner product order results identically. The distance strategy is detected
from the index itself at load time, so no re-embedding is required.


## Acknowledgements

Argo data from the Global Argo Data Repository.
