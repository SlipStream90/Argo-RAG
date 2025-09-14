import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
import pandas as pd
from langchain_community.docstore.in_memory import InMemoryDocstore
import numpy as np
import uuid

chunks=10000
csv_f=r"C:\Users\adity\Desktop\AI_PROJECT\RAG_Setup\argo_preprocessed_with_dates.csv"
def preprocess_chunk(chunk):
    # Combine columns into a single text string per row
    # Modify based on your CSV structure (e.g., select specific columns)
    chunk['text'] = chunk.apply(lambda row: ' '.join(row.astype(str)), axis=1)
    # Create LangChain Documents with metadata
    return [
        Document(page_content=text, metadata={"row_index": i})
        for i, text in enumerate(chunk['text'], start=len(all_texts))
    ]
all_texts = []  # To track processed texts
documents = []

for chunk in pd.read_csv(csv_f, chunksize=chunks):
    chunk_docs = preprocess_chunk(chunk)
    documents.extend(chunk_docs)
    all_texts.extend([doc.page_content for doc in chunk_docs])
    print(f"Processed chunk with {len(chunk_docs)} rows. Total rows processed: {len(documents)}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}  # Use 'cuda' if GPU available
encode_kwargs = {'normalize_embeddings': False}

hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print("Creating embeddings...")
texts = [doc.page_content for doc in documents]
embeds = []
batch_size = 32
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    embeds.extend(hf_embeddings.embed_documents(batch))
embeddings_np = np.array(embeds).astype("float32")
print(f"Embedding array shape: {embeddings_np.shape}")


print("Creating FAISS index...")
dimension = embeddings_np.shape[1]
nlist = min(100, max(1, len(documents) // 10))
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
index.train(embeddings_np)
index.add(embeddings_np)

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
    index_to_docstore_id=index_to_docstore_id
)

print("\nSaving vectorstore...")
try:
    vectorstore.save_local("weather_faiss_vectorstore_main")
    faiss.write_index(index, "faiss_main.bin")
    print("Vectorstore saved successfully!")
except Exception as e:
    print(f"Error saving vectorstore: {str(e)}")
