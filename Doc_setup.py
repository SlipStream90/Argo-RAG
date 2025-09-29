import pandas as pd
from langchain.indexes import VectorstoreIndexCreator
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from langchain_community.docstore.in_memory import InMemoryDocstore
from transformers import pipeline, AutoTokenizer
from langchain_community.vectorstores import FAISS
import faiss
import numpy as np
import uuid

# Preprocess data to reduce volume and sample 1,000 rows
def preprocess_weather_data(df):
    """Summarize data and sample 1,000 rows randomly"""
    # Sample 1,000 rows randomly
    df = df.sample(n=6000, random_state=42)  # Remove random_state for true randomness
    print(f"Sampled dataset size: {len(df)} rows")
    
    # Select relevant columns
    relevant_columns = [col for col in df.columns if 'date' in col.lower() or 'rain' in col.lower() or 'precip' in col.lower() or 'temp' in col.lower()]
    if not relevant_columns:
        relevant_columns = df.columns[:2]
    df = df[relevant_columns]
    
    # Convert date to datetime if needed
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    if date_cols:
        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')
    
    # Aggregate by date (max precipitation per day)
    rain_cols = [col for col in df.columns if 'rain' in col.lower() or 'precip' in col.lower()]
    if rain_cols:
        df = df.groupby(date_cols[0])[rain_cols].max().reset_index()
    
    return df

# Load and preprocess data
print("Loading and preprocessing data...")
df = pd.read_csv("merged_data_1.csv")
df = preprocess_weather_data(df)
print(f"Preprocessed dataset size: {len(df)} rows")

def chunk_dataframe(df, chunk_size=5, max_tokens=150):
    """Convert DataFrame into Document chunks with strict token limit"""
    docs = []
    num_chunks = (len(df) + chunk_size - 1) // chunk_size
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    
    for i in range(num_chunks):
        chunk_df = df.iloc[i*chunk_size : (i+1)*chunk_size]
        relevant_columns = [col for col in chunk_df.columns if 'date' in col.lower() or 'rain' in col.lower() or 'precip' in col.lower()]
        if not relevant_columns:
            relevant_columns = chunk_df.columns[:2]
        
        text_parts = []
        for _, row in chunk_df.iterrows():
            row_text = ", ".join([f"{col}: {row[col]}" for col in relevant_columns if pd.notna(row[col])])
            row_text = row_text[:30]
            text_parts.append(row_text)
        
        text = "\n".join(text_parts)
        tokens = tokenizer(text, return_tensors="pt").input_ids
        if tokens.shape[1] > max_tokens:
            text = text[:int(len(text) * (max_tokens / tokens.shape[1]))]
            tokens = tokenizer(text, return_tensors="pt").input_ids
            print(f"Truncated chunk {i} to {tokens.shape[1]} tokens")
        
        docs.append(Document(page_content=text, metadata={"chunk_index": i, "row_count": len(chunk_df)}))
    
    return docs

# Create document chunks
print("Creating document chunks...")
docs = chunk_dataframe(df, chunk_size=5, max_tokens=150)
print(f"Created {len(docs)} document chunks")

# Initialize embeddings model
print("Initializing embeddings model...")
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}  # Use 'cuda' if GPU available
encode_kwargs = {'normalize_embeddings': False}

hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Create embeddings in batches
print("Creating embeddings...")
texts = [doc.page_content for doc in docs]
embeds = []
batch_size = 20
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    embeds.extend(hf_embeddings.embed_documents(batch))
embeddings_np = np.array(embeds).astype("float32")
print(f"Embedding array shape: {embeddings_np.shape}")

# Create FAISS index
print("Creating FAISS index...")
dimension = embeddings_np.shape[1]
nlist = min(100, max(1, len(docs) // 10))
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
index.train(embeddings_np)
index.add(embeddings_np)

# Create docstore
docstore = InMemoryDocstore()
index_to_docstore_id = {}
for i, doc in enumerate(docs):
    doc_id = str(uuid.uuid4())
    docstore.add({doc_id: doc})
    index_to_docstore_id[i] = doc_id

# Create FAISS vectorstore
vectorstore = FAISS(
    embedding_function=hf_embeddings,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id
)

# Create retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# Initialize language model
print("Initializing language model...")
hf_pipeline = pipeline(
    "text2text-generation", 
    model="google/flan-t5-base",
    max_length=512,
    do_sample=False
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Define map and combine prompts for map_reduce

# Create QA chain with map_reduce
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
)

# Test the system
print("\nTesting the RAG system...")
queries = [
    "Which date had the highest rainfall?",
    "What was the average temperature?",
    "Tell me about average wind speed in Jan 2024?",
]

for query in queries:
    print(f"\nQuery: {query}")
    try:
        result = qa_chain.invoke({"query": query})
        print(f"Answer: {result['result']}")
        print(f"Number of source documents: {len(result['source_documents'])}")
    except Exception as e:
        print(f"Error: {str(e)}")

# Save the vectorstore
print("\nSaving vectorstore...")
try:
    vectorstore.save_local("weather_faiss_vectorstore")
    faiss.write_index(index, "faiss_index.bin")
    print("Vectorstore saved successfully!")
except Exception as e:
    print(f"Error saving vectorstore: {str(e)}")

print("\nRAG system setup complete!")