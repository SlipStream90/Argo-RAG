"""Query-time RAG pipeline over the Argo float corpus.

Hybrid retrieval (dense FAISS + lexical BM25, fused with weighted RRF) feeding
a local Ollama model through an LCEL chain.

Public entry point, consumed by app.py:

    main(query) -> (answer, num_source_documents)

Everything is built lazily on first query, so importing this module -- which
Streamlit does at start-up -- no longer blocks for minutes loading a 227 MB
index, and no longer crashes the whole app if the index is missing.
"""

from __future__ import annotations

import argparse
import re
import sys
from functools import lru_cache
from typing import List, Tuple

import requests
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

import config
from retrieval import (
    DateAwareRetriever,
    FusionRetriever,
    VectorstoreMissingError,
    build_bm25,
    iter_documents,
    load_embeddings,
    load_vectorstore,
    looks_like_aggregate,
)


class LLMUnavailableError(RuntimeError):
    """Raised when the Ollama server cannot be reached."""


# ──────────────────────────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────────────────────────

def _field_reference() -> str:
    """Describe the fields the corpus actually contains.

    Generated from config.FIELD_LABELS rather than hand-written: the previous
    prompt advertised Depth and Quality Flag columns that do not exist in this
    dataset, inviting the model to invent them.
    """
    return "\n".join(
        f"  {label} ({unit})" for label, unit in config.FIELD_LABELS.values()
    )


PROMPT_TEMPLATE = """You are an expert oceanographer analysing Argo float data.
Each retrieved record is one profile measurement, formatted as labelled pairs:
  Field Name: value | Field Name: value | ...

Fields present in this dataset:
{fields}

Retrieved records:
{context}

Question: {question}
{caveat}
Instructions:
- Answer only from the records above; never invent values or fields.
- If the exact date is unavailable, use the nearest record and state its actual
  date and the offset in days.
- Quote measurements with their units and include the coordinates.
- If the records do not support an answer, say so plainly.
- Be concise and direct.

Answer:"""

AGGREGATE_CAVEAT = """
IMPORTANT: this question asks for a statistic across the dataset, but the
records above are only the top matches, not the full corpus. Do not compute an
average, total or extreme from them as if it covered everything. Report what
these specific records show and state clearly that a corpus-wide statistic
cannot be derived from a retrieval sample.
"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question", "caveat", "fields"],
)


# ──────────────────────────────────────────────────────────────────
# LLM
# ──────────────────────────────────────────────────────────────────

def _call_ollama(prompt: str) -> str:
    """Generate with Ollama over plain HTTP.

    Avoids the langchain-ollama dependency for what is one POST, and lets us
    turn a connection refusal into an explanatory error instead of a traceback.
    """
    try:
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": config.LLM_NUM_PREDICT,
                    "temperature": config.LLM_TEMPERATURE,
                    "top_p": config.LLM_TOP_P,
                    "num_ctx": config.LLM_NUM_CTX,
                },
            },
            timeout=config.LLM_TIMEOUT,
        )
    except requests.exceptions.ConnectionError as exc:
        raise LLMUnavailableError(
            f"Cannot reach Ollama at {config.OLLAMA_BASE_URL}. "
            f"Start it with 'ollama serve' and pull the model with "
            f"'ollama pull {config.OLLAMA_MODEL}'."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise LLMUnavailableError(
            f"Ollama did not respond within {config.LLM_TIMEOUT}s."
        ) from exc

    if response.status_code == 404:
        raise LLMUnavailableError(
            f"Model '{config.OLLAMA_MODEL}' is not installed. "
            f"Run: ollama pull {config.OLLAMA_MODEL}"
        )
    response.raise_for_status()
    return strip_reasoning(response.json().get("response", ""))


# qwen3 and other reasoning models emit their chain of thought in <think>
# blocks. Left in, it reaches the Streamlit UI verbatim and buries the answer.
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK = re.compile(r"^.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove <think> blocks from a model response."""
    cleaned = _THINK_BLOCK.sub("", text)
    # A response truncated by num_predict can open <think> and never close it;
    # if a stray closing tag remains, drop everything up to it.
    if "</think>" in cleaned:
        cleaned = _UNCLOSED_THINK.sub("", cleaned)
    return cleaned.strip()


# ──────────────────────────────────────────────────────────────────
# Chain construction (lazy, cached)
# ──────────────────────────────────────────────────────────────────

def format_docs(docs: List[Document]) -> str:
    return "\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, 1))


@lru_cache(maxsize=1)
def get_retriever(use_bm25: bool = True):
    """Build the hybrid retriever once and reuse it."""
    embeddings = load_embeddings()
    store = load_vectorstore(embeddings)

    dense = DateAwareRetriever(
        base_retriever=store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": config.TOP_K,
                "fetch_k": config.FETCH_K,
                "lambda_mult": config.MMR_LAMBDA,
            },
        )
    )

    if not use_bm25:
        return dense

    lexical = DateAwareRetriever(
        base_retriever=build_bm25(iter_documents(store), k=config.TOP_K)
    )

    return FusionRetriever(
        retrievers=[dense, lexical],
        weights=[config.DENSE_WEIGHT, config.LEXICAL_WEIGHT],
        rrf_k=config.RRF_K,
        top_k=config.TOP_K,
    )


@lru_cache(maxsize=1)
def get_chain(use_bm25: bool = True):
    """LCEL chain returning both the answer and the documents it used.

    Replaces RetrievalQA, which lived in `langchain.chains` -- removed in
    langchain 1.x, so the previous module could not even be imported here.
    """
    retriever = get_retriever(use_bm25)

    answer = (
        {
            "context": lambda x: format_docs(x["documents"]),
            "question": lambda x: x["question"],
            "caveat": lambda x: AGGREGATE_CAVEAT if looks_like_aggregate(x["question"]) else "",
            "fields": lambda _: _field_reference(),
        }
        | PROMPT
        | RunnableLambda(_call_ollama)
        | StrOutputParser()
    )

    return {
        "question": RunnablePassthrough(),
        "documents": retriever,
    } | RunnableParallel(answer=answer, documents=lambda x: x["documents"])


# ──────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────

def show_retrieved_docs(docs: List[Document]) -> None:
    print("\n" + "=" * 60)
    print(f"RETRIEVED DOCUMENTS ({len(docs)} total)")
    print("=" * 60)
    for i, doc in enumerate(docs, 1):
        print(f"\n[{i}] Metadata: {doc.metadata}")
        print(f"    Content:  {doc.page_content}")
        print("-" * 40)


def run_query(query: str, verbose: bool = True, use_bm25: bool = True):
    result = get_chain(use_bm25).invoke(query)
    docs = result["documents"]
    if verbose:
        show_retrieved_docs(docs)
    return result["answer"], len(docs), docs


def main(query: str) -> Tuple[str, int]:
    """Entry point used by app.py. Errors come back as text, not exceptions,
    so a missing index or a stopped Ollama shows up in the UI as a message."""
    try:
        answer, num_docs, _ = run_query(query, verbose=False)
    except (VectorstoreMissingError, LLMUnavailableError) as exc:
        return f"Cannot answer this query: {exc}", 0
    return answer, num_docs


def preflight() -> int:
    """Report exactly which prerequisites are missing, with the fix for each.

    Beats discovering a missing package as a traceback from inside FAISS
    deserialisation, or a stopped Ollama as a connection error mid-answer.
    """
    import importlib.util
    from pathlib import Path

    problems = []

    for module, fix in [
        ("langchain_community", "pip install -r requirements.txt"),
        ("langchain_huggingface", "pip install -r requirements.txt"),
        ("rank_bm25", "pip install rank-bm25"),
        ("faiss", "pip install faiss-cpu"),
        ("dateparser", "pip install dateparser"),
    ]:
        ok = importlib.util.find_spec(module) is not None
        print(f"  [{'ok ' if ok else 'MISSING'}] {module}")
        if not ok:
            problems.append(f"{module}: {fix}")

    index = Path(config.VECTORSTORE_PATH) / "index.faiss"
    if index.exists():
        print(f"  [ok ] index ({index.stat().st_size / 1e6:.0f} MB)")
    else:
        print("  [MISSING] FAISS index")
        problems.append(f"index: python embed_gen.py --csv {config.CSV_PATH}")

    try:
        tags = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5).json()
        models = [m["name"] for m in tags.get("models", [])]
        if config.OLLAMA_MODEL in models:
            print(f"  [ok ] ollama, model {config.OLLAMA_MODEL}")
        else:
            print(f"  [MISSING] ollama model {config.OLLAMA_MODEL} (have: {models})")
            problems.append(f"model: ollama pull {config.OLLAMA_MODEL}")
    except Exception:
        print(f"  [MISSING] ollama at {config.OLLAMA_BASE_URL}")
        problems.append("ollama: start it with 'ollama serve'")

    if problems:
        print("\nTo fix:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nAll checks passed.")
    return 0


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Query the Argo RAG pipeline.")
    parser.add_argument("query", nargs="*", help="question to ask")
    parser.add_argument("--check", action="store_true", help="verify prerequisites and exit")
    parser.add_argument("-q", "--quiet", action="store_true", help="hide retrieved documents")
    parser.add_argument("--no-bm25", action="store_true", help="dense retrieval only")
    parser.add_argument("-k", "--top-k", type=int, help="number of documents to retrieve")
    args = parser.parse_args()

    if args.check:
        return preflight()

    if args.top_k:
        config.TOP_K = args.top_k

    question = " ".join(args.query) or input("Question: ").strip()
    if not question:
        parser.error("no question given")

    try:
        answer, num_docs, _ = run_query(
            question, verbose=not args.quiet, use_bm25=not args.no_bm25
        )
    except (VectorstoreMissingError, LLMUnavailableError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"\nAnswer ({num_docs} docs used):\n{answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
