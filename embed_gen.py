"""Build the FAISS vectorstore from the preprocessed Argo CSV.

    python embed_gen.py                     # build with defaults from config.py
    python embed_gen.py --limit 5000        # quick smoke build
    python embed_gen.py --dry-run           # render documents, embed nothing

One CSV row becomes one Document. page_content is a set of labelled
"Field: value" pairs so the embedder sees what each number means; metadata
keeps the raw typed values so retrieval results can be filtered or plotted
without re-parsing the text.
"""

from __future__ import annotations

import argparse
import math
import re
import uuid
from pathlib import Path
from typing import List

import faiss
import numpy as np
import pandas as pd
from langchain_core.documents import Document

# langchain_community is imported inside build_vectorstore() so that --dry-run
# and document rendering work with pandas alone.

import config
from retrieval import build_bm25, load_embeddings

# Several columns arrive as numpy bytes reprs, e.g. "b'1902029 '". Left as-is
# they leak Python syntax into the embedded text and into the LLM's context.
_BYTES_REPR = re.compile(r"^b['\"](.*)['\"]$")


def _clean(value: object) -> str:
    text = str(value).strip()
    match = _BYTES_REPR.match(text)
    return match.group(1).strip() if match else text


def _render(value: object) -> str:
    """Format a value for the embedded text.

    Binary floats stringify with their full repr ("47.650120000000015"), which
    is noise the embedder and the LLM both have to read. Five decimal places is
    about a metre of position -- far beyond what this data resolves.
    """
    if isinstance(value, float):
        return f"{round(value, 5):g}"
    return _clean(value)


def row_to_document(row: pd.Series, row_index: int) -> Document:
    columns = row.index.tolist()
    ordered = [c for c in config.PRIORITY_COLS if c in columns] + sorted(
        c for c in columns
        if c not in config.PRIORITY_COLS and c.lower() not in config.SKIP_COLS
    )

    parts: List[str] = []
    metadata: dict = {"row_index": row_index}

    for col in ordered:
        raw = row[col]
        if pd.isna(raw):
            continue
        value = _render(raw)
        if value in ("", "nan", "NaT", "None"):
            continue

        label = config.FIELD_LABELS.get(col, (col.replace("_", " ").title(), ""))[0]
        parts.append(f"{label}: {value}")

        # Numpy scalars do not pickle back as plain Python types; convert so the
        # metadata stays usable (and JSON-serialisable) downstream. Strings get
        # the same bytes-repr cleanup as the text, so a metadata filter on
        # platform_number matches what the user actually sees.
        typed = raw.item() if hasattr(raw, "item") else raw
        metadata[col] = _clean(typed) if isinstance(typed, (str, bytes)) else typed

    return Document(page_content=" | ".join(parts), metadata=metadata)


def load_documents(csv_path: Path, limit: int | None = None) -> List[Document]:
    if not csv_path.exists():
        raise SystemExit(f"error: CSV not found at '{csv_path}'")

    documents: List[Document] = []
    for chunk in pd.read_csv(csv_path, chunksize=config.CSV_CHUNK_SIZE):
        chunk.columns = chunk.columns.str.strip().str.lower()
        offset = len(documents)
        documents.extend(
            row_to_document(row, row_index=offset + i)
            for i, (_, row) in enumerate(chunk.iterrows())
        )
        print(f"  loaded {len(documents):,} rows")
        if limit and len(documents) >= limit:
            del documents[limit:]
            break

    if not documents:
        raise SystemExit(f"error: '{csv_path}' produced no rows")
    return documents


def embed_documents(documents: List[Document]) -> np.ndarray:
    """Embed in batches, accumulating float32 arrays rather than Python lists.

    A list of hundreds of thousands of 384-float Python lists costs several
    gigabytes before np.array() ever sees it.
    """
    embeddings = load_embeddings()
    texts = [doc.page_content for doc in documents]
    batches: List[np.ndarray] = []

    for start in range(0, len(texts), config.EMBED_BATCH_SIZE):
        batch = texts[start : start + config.EMBED_BATCH_SIZE]
        batches.append(np.asarray(embeddings.embed_documents(batch), dtype="float32"))
        done = min(start + config.EMBED_BATCH_SIZE, len(texts))
        if (start // config.EMBED_BATCH_SIZE) % 50 == 0 or done == len(texts):
            print(f"  embedded {done:,}/{len(texts):,}")

    return np.vstack(batches)


def build_index(vectors: np.ndarray) -> faiss.Index:
    """IVF index over inner product.

    The embeddings are L2-normalised, so inner product is cosine similarity.
    (On unit vectors an L2 index ranks identically -- the previous IndexFlatL2
    quantizer was not returning wrong neighbours -- but with inner product the
    scores themselves are meaningful similarities in [-1, 1] rather than
    squared distances.)
    """
    n, dim = vectors.shape

    # FAISS wants >= ~39 training points per centroid; nlist ~ sqrt(n)
    # otherwise. Too many centroids on a small corpus triggers warnings and
    # poor recall.
    nlist = max(1, min(config.IVF_NLIST_CAP, int(math.sqrt(n)), n // 39 or 1))

    if config.INDEX_TYPE == "flat" or nlist == 1:
        # Exact search. At ~150k x 384 this is tens of milliseconds per query,
        # so IVF's approximation buys nothing while adding two tuning knobs.
        print(f"  flat index (exact) | vectors={n:,}")
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        return index

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    index.add(vectors)
    index.nprobe = config.NPROBE
    print(f"  IVF index | nlist={nlist} nprobe={index.nprobe} vectors={index.ntotal:,}")
    return index


def build_vectorstore(documents: List[Document], index: faiss.Index):
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy

    docstore = InMemoryDocstore()
    index_to_docstore_id = {}
    for i, doc in enumerate(documents):
        doc_id = str(uuid.uuid4())
        docstore.add({doc_id: doc})
        index_to_docstore_id[i] = doc_id

    return FAISS(
        embedding_function=load_embeddings(),
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=config.CSV_PATH)
    parser.add_argument("--out", type=Path, default=config.VECTORSTORE_PATH)
    parser.add_argument("--limit", type=int, help="only index the first N rows")
    parser.add_argument("--dry-run", action="store_true", help="render documents only")
    parser.add_argument("--no-bm25-cache", action="store_true", help="skip BM25 prebuild")
    args = parser.parse_args()

    print(f"Reading {args.csv}")
    documents = load_documents(args.csv, args.limit)

    print("\n-- sample document --")
    print(documents[0].page_content)
    print("metadata:", documents[0].metadata)

    if args.dry_run:
        print(f"\nDry run: {len(documents):,} documents rendered, nothing embedded.")
        return 0

    print(f"\nEmbedding {len(documents):,} documents ({config.EMBED_MODEL})")
    vectors = embed_documents(documents)
    print(f"  shape {vectors.shape}")

    print("\nBuilding FAISS index")
    index = build_index(vectors)

    print(f"\nSaving to {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    build_vectorstore(documents, index).save_local(str(args.out))

    if not args.no_bm25_cache:
        # Pay the tokenisation cost once here rather than on the first query.
        print("Prebuilding BM25 cache")
        build_bm25(documents, k=config.TOP_K)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
