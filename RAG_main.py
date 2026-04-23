import re
import dateparser
from dateparser.search import search_dates
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from pydantic import Field
from typing import List

hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = FAISS.load_local(
    folder_path="weather_faiss_vectorstore_main",
    embeddings=hf_embeddings,
    allow_dangerous_deserialization=True,
)
vectorstore.index.nprobe = 10

_FALSE_POSITIVE_PATTERNS = [
    r"\b\d+\s*(meters?|m|km|PSU|°C|dbar|knots?)\b",
    r"\b(station|platform|float|id)\s*\d+\b",
]

def _is_false_positive(text: str, span: str) -> bool:
    for pattern in _FALSE_POSITIVE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def normalise_dates_in_query(query: str) -> str:
    found = search_dates(
        query,
        languages=["en"],
        settings={
            "PREFER_DAY_OF_MONTH": "first",
            "RETURN_TIME_AS_PERIOD": False,
            "PREFER_DATES_FROM": "past",
        },
    )

    if not found:
        return query

    result = query
    replacements = []

    for original_text, parsed_dt in found:
        if _is_false_positive(query, original_text):
            continue
        if re.fullmatch(r"-?\d+(\.\d+)?", original_text.strip()):
            continue
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", original_text.strip()):
            continue

        iso_date = parsed_dt.strftime("%Y-%m-%d")
        replacements.append((original_text, iso_date))

    replacements.sort(key=lambda x: len(x[0]), reverse=True)
    for original_text, iso_date in replacements:
        result = result.replace(original_text, iso_date, 1)

    if result != query:
        print(f"[DateParser] '{query}'  →  '{result}'")

    return result


class DateAwareRetriever(BaseRetriever):
    base_retriever: BaseRetriever = Field(...)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        clean_query = normalise_dates_in_query(query)
        return self.base_retriever.invoke(clean_query)


all_documents: List[Document] = list(vectorstore.docstore._dict.values())

bm25_base = BM25Retriever.from_documents(all_documents, k=6)

faiss_base = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7},
)

bm25_retriever  = DateAwareRetriever(base_retriever=bm25_base)
faiss_retriever = DateAwareRetriever(base_retriever=faiss_base)

ensemble_retriever = EnsembleRetriever(
    retrievers=[faiss_retriever, bm25_retriever],
    weights=[0.5, 0.5],
)

llm = OllamaLLM(
    model="qwen3:latest",
    base_url="http://localhost:11434",
    num_predict=2048,
    temperature=0.1,
    top_p=0.75,
)

PROMPT_TEMPLATE = """You are an expert oceanographer analysing Argo float data.
Each retrieved record is formatted as labelled pairs:
  Field Name: value | Field Name: value | ...

Common fields and their units:
  Date (YYYY-MM-DD), Latitude / Longitude (decimal degrees, negative = S/W),
  Depth (m), Pressure (dbar), Temperature (°C), Salinity (PSU),
  Quality Flag (0 = good), Platform / Station ID

Retrieved records:
{context}

Question: {question}

Instructions:
- Answer only from the records above; do not hallucinate values.
- If the exact date is unavailable use the nearest record and state the actual date and day offset.
- Quote specific measurements with units and include coordinates.
- Be concise and direct.

Answer:"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=ensemble_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
)


def show_retrieved_docs(docs: list) -> None:
    print("\n" + "=" * 60)
    print(f"RETRIEVED DOCUMENTS ({len(docs)} total)")
    print("=" * 60)
    for i, doc in enumerate(docs, 1):
        print(f"\n[{i}] Metadata: {doc.metadata}")
        print(f"    Content:  {doc.page_content}")
        print("-" * 40)


def run_query(query: str, verbose: bool = True):
    result = qa_chain.invoke({"query": query})
    answer = result["result"]
    source_docs = result["source_documents"]
    if verbose:
        show_retrieved_docs(source_docs)
    return answer, len(source_docs), source_docs


def main(query: str) -> tuple[str, int]:
    answer, num_docs, _ = run_query(query, verbose=True)
    print(f"\nAnswer ({num_docs} docs used):\n{answer}")
    return answer, num_docs


if __name__ == "__main__":
    main("What was the temperature on January 3rd, 2023 near 45N 30W?")
