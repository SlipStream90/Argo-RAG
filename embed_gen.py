import faiss
import numpy as np
import uuid
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

CSV_PATH = r"C:\Users\adity\Desktop\AI_PROJECT\RAG_Setup\argo_preprocessed_with_dates.csv"
CHUNK_SIZE = 10_000
BATCH_SIZE = 32
SAVE_PATH = "weather_faiss_vectorstore_main"

# ──────────────────────────────────────────────
# 1. Row → human-readable document
# ──────────────────────────────────────────────

# Columns that carry rich semantic meaning and should lead the document.
# Adjust to match your actual CSV headers.
PRIORITY_COLS = ["date", "region", "parameter", "depth", "value", "unit", "quality_flag"]

# Columns that are pure identifiers / indices — keep as metadata only.
SKIP_COLS = {"row_id", "index", "unnamed: 0"}

def row_to_document(row: pd.Series, row_index: int) -> Document:
    """
    Convert a single CSV row into a LangChain Document.

    page_content  →  labeled field: value pairs so the embedder
                     understands what each number means.
    metadata      →  raw typed values for post-retrieval filtering.
    """
    columns = row.index.tolist()

    # Sort so priority cols appear first, then everything else alphabetically
    ordered_cols = [c for c in PRIORITY_COLS if c in columns] + \
                   sorted([c for c in columns
                           if c not in PRIORITY_COLS and c.lower() not in SKIP_COLS])

    parts = []
    metadata = {"row_index": row_index}

    for col in ordered_cols:
        raw = row[col]

        # Skip NaN / empty
        if pd.isna(raw) or str(raw).strip() in ("", "nan", "NaT"):
            continue

        value = str(raw).strip()

        # Human-readable label: "sea_surface_temp" → "Sea Surface Temp"
        label = col.replace("_", " ").title()
        parts.append(f"{label}: {value}")

        # Preserve typed value in metadata for filtering
        metadata[col] = raw

    page_content = " | ".join(parts)
    return Document(page_content=page_content, metadata=metadata)


# ──────────────────────────────────────────────
# 2. Load CSV in chunks
# ──────────────────────────────────────────────
documents = []

for chunk in pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE):
    offset = len(documents)

    # Normalise column names once per chunk
    chunk.columns = chunk.columns.str.strip().str.lower()

    chunk_docs = [
        row_to_document(row, row_index=offset + i)
        for i, (_, row) in enumerate(chunk.iterrows())
    ]
    documents.extend(chunk_docs)
    print(f"Loaded {len(chunk_docs)} rows | Total: {len(documents)}")

# Sanity-check: print the first document so you can verify the format
print("\n── Sample document ──")
print(documents[0].page_content)
print("Metadata:", documents[0].metadata)

# ──────────────────────────────────────────────
# 3. Embeddings
# ──────────────────────────────────────────────
hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print("\nCreating embeddings...")
texts = [doc.page_content for doc in documents]
embeds = []

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i : i + BATCH_SIZE]
    embeds.extend(hf_embeddings.embed_documents(batch))
    if (i // BATCH_SIZE) % 50 == 0:
        print(f"  Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")

embeddings_np = np.array(embeds, dtype="float32")
print(f"Embedding shape: {embeddings_np.shape}")

# ──────────────────────────────────────────────
# 4. FAISS index
# ──────────────────────────────────────────────
print("\nBuilding FAISS index...")
dim = embeddings_np.shape[1]
nlist = min(100, max(1, len(documents) // 10))

quantizer = faiss.IndexFlatL2(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist)
index.train(embeddings_np)
index.add(embeddings_np)
index.nprobe = 10  # probe 10 clusters at query time for better recall
print(f"Index built | nlist={nlist}, nprobe={index.nprobe}, vectors={index.ntotal}")

# ──────────────────────────────────────────────
# 5. Docstore + vectorstore
# ──────────────────────────────────────────────
docstore = InMemoryDocstore()
index_to_docstore_id = {}

for i, doc in enumerate(documents):
    doc_id = str(uuid.uuid4())
    docstore.add({doc_id: doc})
    index_to_docstore_id[i] = doc_id

vectorstore = FAISS(
    embedding_function=hf_embeddings,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id,
)

print("\nSaving vectorstore...")
try:
    vectorstore.save_local(SAVE_PATH)
    print(f"Saved to '{SAVE_PATH}/'")
except Exception as e:
    print(f"Save failed: {e}")
