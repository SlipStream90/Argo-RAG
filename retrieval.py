"""Retrieval layer for the Argo RAG pipeline.

Split out of RAG_main so the query-rewriting and fusion logic can be unit
tested without loading a 227 MB FAISS index or starting an LLM.

Heavy dependencies (faiss, langchain_community) are imported inside the
functions that need them; importing this module is cheap.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Iterable, List, Sequence

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

import config


class VectorstoreMissingError(RuntimeError):
    """Raised when the FAISS artefacts are absent, with a build hint."""


# ──────────────────────────────────────────────────────────────────
# Date normalisation
# ──────────────────────────────────────────────────────────────────

# A span that looks like one of these is a measurement or an identifier, not a
# date. dateparser happily reads "30 m" as "the 30th of the current month", so
# these spans must be rejected before they corrupt the query.
_FALSE_POSITIVE_PATTERNS = [
    re.compile(r"^\s*-?\d+(\.\d+)?\s*(meters?|metres?|m|km|psu|dbar|knots?|deg(rees)?|c|°c)\s*$", re.I),
    re.compile(r"^\s*(station|platform|float|cycle|profile|id)\s*#?\s*\d+\s*$", re.I),
    re.compile(r"^\s*-?\d+(\.\d+)?\s*°?\s*[nsew]\s*$", re.I),  # 45N, 30.5 W
]

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_BARE_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")

# dateparser returns spans with the preposition attached ("on January 3rd"),
# which would otherwise be swallowed by the replacement.
_LEADING_CONNECTIVE = re.compile(
    r"^(on|at|in|from|since|during|by|for|of|the)\s+", re.I
)

# Measurements and coordinates anywhere in the query derail dateparser's
# tokeniser: in "temperature at 10 m on January 3rd, 2023" it gives up on the
# real date and returns a bare "2023". Blanking them out first (preserving
# length, so every other offset is untouched) lets the real date parse.
_MEASUREMENT_IN_TEXT = re.compile(
    r"\b-?\d+(\.\d+)?\s*(meters?|metres?|m|km|psu|dbar|knots?|°c|°|c)\b"
    r"|\b-?\d+(\.\d+)?\s*°?\s*[nsew]\b"
    r"|\b(station|platform|float|cycle|profile|id)\s*#?\s*\d+\b",
    re.I,
)


def _mask_measurements(query: str) -> str:
    return _MEASUREMENT_IN_TEXT.sub(lambda m: " " * len(m.group(0)), query)


def _tighten_span(span: str, query: str) -> str | None:
    """Reduce a dateparser span to the shortest suffix that is real text.

    Two things make the raw span unusable. It carries prepositions ("on
    January 3rd"), and because masking can leave two connectives adjacent,
    dateparser may hand back text that never occurs in the original query at
    all ("at on January 3rd, 2023"). Dropping leading tokens until the
    remainder is a genuine substring handles both, and returns None when
    nothing matches rather than corrupting the query.
    """
    span = span.strip()
    while span:
        if _LEADING_CONNECTIVE.match(span):
            span = _LEADING_CONNECTIVE.sub("", span, count=1).strip()
            continue
        if span in query:
            return span
        # Not a substring: shed the leading token and try again.
        head, _, tail = span.partition(" ")
        if not tail:
            return None
        span = tail.strip()
    return None


def is_false_positive(span: str) -> bool:
    """True if `span` is a measurement/identifier rather than a real date.

    Note this inspects the matched span only. Testing the whole query (the
    previous behaviour) meant a single "10 m" anywhere disabled date parsing
    for every span in that query.
    """
    return any(p.match(span) for p in _FALSE_POSITIVE_PATTERNS)


def normalise_dates_in_query(query: str, verbose: bool = False) -> str:
    """Rewrite natural-language dates to ISO so they match the indexed text.

    Documents store dates as ``Date: 2025-01-11``; a query saying "January 3rd
    2023" shares no tokens with that, which hurts the lexical retriever badly
    and the dense one mildly.
    """
    from dateparser.search import search_dates

    masked = _mask_measurements(query)

    try:
        found = search_dates(
            masked,
            languages=["en"],
            settings={
                "PREFER_DAY_OF_MONTH": "first",
                "RETURN_TIME_AS_PERIOD": False,
                "PREFER_DATES_FROM": "past",
            },
        )
    except Exception:
        # dateparser raises on some odd inputs; a failed rewrite must never
        # take down a query.
        return query

    if not found:
        return query

    replacements = []
    for original_text, parsed_dt in found:
        span = _tighten_span(original_text, query)
        if not span or is_false_positive(span):
            continue
        # A bare number ("2023") is dateparser guessing; it would expand a year
        # into an arbitrary day. Already-ISO spans need no rewriting.
        if _BARE_NUMBER.match(span) or _ISO_DATE.match(span):
            continue
        replacements.append((span, parsed_dt.strftime("%Y-%m-%d")))

    # Longest span first, so replacing "January 3rd" cannot chew up part of
    # "January 3rd, 2023".
    replacements.sort(key=lambda pair: len(pair[0]), reverse=True)

    result = query
    for original_text, iso_date in replacements:
        result = result.replace(original_text, iso_date, 1)

    if verbose and result != query:
        print(f"[dates] {query!r} -> {result!r}")
    return result


# ──────────────────────────────────────────────────────────────────
# Aggregate-query detection
# ──────────────────────────────────────────────────────────────────

_AGGREGATE_TERMS = re.compile(
    r"\b(average|avg|mean|median|total|sum|count|how many|highest|lowest|"
    r"maximum|minimum|max|min|trend|over time|distribution)\b",
    re.I,
)


def looks_like_aggregate(query: str) -> bool:
    """True if the question asks for a statistic over the whole corpus.

    Top-k retrieval cannot answer these: it returns 6 rows out of hundreds of
    thousands, so any "average" computed from them is wrong. Detecting the
    intent lets the prompt tell the model to say so instead of inventing a
    confident number from a biased sample.
    """
    return bool(_AGGREGATE_TERMS.search(query))


# ──────────────────────────────────────────────────────────────────
# Retrievers
# ──────────────────────────────────────────────────────────────────


class DateAwareRetriever(BaseRetriever):
    """Rewrites dates in the query, then delegates to `base_retriever`."""

    base_retriever: BaseRetriever = Field(...)
    verbose: bool = False

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.base_retriever.invoke(normalise_dates_in_query(query, self.verbose))


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Document]],
    weights: Sequence[float],
    rrf_k: int = 60,
    top_k: int | None = None,
) -> List[Document]:
    """Fuse several ranked document lists with weighted RRF.

    RRF scores by rank rather than by raw score, which is what makes it safe to
    combine a cosine-similarity list with a BM25 list -- their scores are on
    completely different scales and are not comparable.
    """
    scores: dict[str, float] = {}
    seen: dict[str, Document] = {}

    for docs, weight in zip(ranked_lists, weights):
        for rank, doc in enumerate(docs):
            key = _doc_key(doc)
            seen.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank + 1)

    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    fused = [seen[key] for key in ordered]
    return fused[:top_k] if top_k else fused


def _doc_key(doc: Document) -> str:
    """Stable identity for a document across retrievers."""
    row_index = doc.metadata.get("row_index")
    return f"row:{row_index}" if row_index is not None else f"txt:{doc.page_content}"


class FusionRetriever(BaseRetriever):
    """Weighted RRF over several retrievers.

    Replaces langchain's EnsembleRetriever, which lives in `langchain.retrievers`
    -- a module removed in langchain 1.x. Same algorithm, one less dependency.
    """

    retrievers: List[BaseRetriever] = Field(...)
    weights: List[float] = Field(...)
    rrf_k: int = config.RRF_K
    top_k: int = config.TOP_K

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        ranked = [r.invoke(query) for r in self.retrievers]
        return reciprocal_rank_fusion(ranked, self.weights, self.rrf_k, self.top_k)


# ──────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────


def load_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        model_kwargs={"device": config.EMBED_DEVICE},
        encode_kwargs={"normalize_embeddings": config.NORMALIZE_EMBEDDINGS},
    )


def load_vectorstore(embeddings=None):
    """Load the saved FAISS store, failing with an actionable message."""
    from langchain_community.vectorstores import FAISS

    path = Path(config.VECTORSTORE_PATH)
    if not (path / "index.faiss").exists():
        raise VectorstoreMissingError(
            f"No FAISS index at '{path}'.\n"
            f"Build one first:  python embed_gen.py --csv {config.CSV_PATH}"
        )

    store = FAISS.load_local(
        folder_path=str(path),
        embeddings=embeddings or load_embeddings(),
        allow_dangerous_deserialization=True,
    )

    _apply_distance_strategy(store)

    _set_nprobe(store)
    return store


def _set_nprobe(store) -> None:
    """Apply nprobe to the IVF layer, if there is one.

    `store.index` is the raw faiss index, so assigning to it does reach FAISS
    -- but only when the index is the IVF itself. If FAISS returns a wrapper
    (IndexPreTransform, IndexIDMap), SWIG happily accepts `.nprobe = 10` as a
    dead Python attribute that never affects the search. extract_index_ivf
    unwraps to the real IVF, and raises when there is none (a flat index, which
    needs no nprobe at all).
    """
    import faiss

    try:
        faiss.extract_index_ivf(store.index).nprobe = config.NPROBE
    except RuntimeError:
        pass  # flat index: exact search, nothing to probe


def _apply_distance_strategy(store) -> None:
    """Match langchain's distance strategy to the index's actual metric.

    save_local() persists only the docstore and the id map, so the metric is
    not carried with it and load_local() always assumes Euclidean. Reading the
    metric off the loaded index keeps both the original L2 build and a new
    inner-product build scoring correctly, with no flag for a caller to get
    wrong. Ranking is unaffected either way on normalised vectors; what this
    fixes is the score scale that MMR's lambda_mult and any relevance
    threshold are interpreted against.
    """
    import faiss
    from langchain_community.vectorstores.utils import DistanceStrategy

    if getattr(store.index, "metric_type", None) == faiss.METRIC_INNER_PRODUCT:
        store.distance_strategy = DistanceStrategy.MAX_INNER_PRODUCT


def iter_documents(store) -> List[Document]:
    """All documents in the store, for building the lexical index.

    langchain exposes no public iteration API over InMemoryDocstore, so this
    reaches into `_dict` -- but it does so in exactly one place, and degrades
    with a clear error rather than an AttributeError deep in a retriever.
    """
    docstore = getattr(store, "docstore", None)
    mapping = getattr(docstore, "_dict", None)
    if mapping is None:
        raise RuntimeError(
            "Cannot enumerate documents: this docstore has no `_dict`. "
            "BM25 retrieval is unavailable; run with --no-bm25."
        )
    return list(mapping.values())


def build_bm25(documents: Iterable[Document], k: int = config.TOP_K, cache: bool = True):
    """BM25 retriever, cached to disk so start-up does not re-tokenise.

    Tokenising several hundred thousand documents takes tens of seconds on
    every single run; the cache turns that into a pickle load.
    """
    from langchain_community.retrievers import BM25Retriever

    cache_path = Path(config.BM25_CACHE_PATH)
    if cache and cache_path.exists():
        try:
            with cache_path.open("rb") as fh:
                retriever = pickle.load(fh)
            retriever.k = k
            return retriever
        except Exception as exc:  # corrupt or version-mismatched cache
            print(f"[bm25] ignoring unusable cache ({exc}); rebuilding")

    retriever = BM25Retriever.from_documents(list(documents), k=k)

    if cache:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as fh:
                pickle.dump(retriever, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            print(f"[bm25] could not write cache: {exc}")

    return retriever
