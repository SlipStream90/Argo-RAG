# RAG_Setup

Retrieval-Augmented Generation (RAG) for Oceanographic Data Analysis

## Overview

This project implements a RAG pipeline using FAISS for vector search, HuggingFace and Ollama for LLMs, and custom prompt engineering for oceanographic data. It enables users to query marine datasets and receive expert-level answers with context-aware retrieval.

## Features

- **Vector Search:** Uses FAISS for fast similarity search over embedded documents.
- **Embeddings:** Supports HuggingFace and SentenceTransformers for generating document embeddings.
- **LLM Integration:** Uses Ollama and HuggingFace pipelines for natural language generation.
- **Custom Prompt:** Tailored for oceanographic data, focusing on measurements, dates, and locations.
- **Streamlit UI:** (if included) for interactive querying.

## File Structure

- `RAG_main.py` — Main RAG pipeline and prompt logic.
- `embed_gen.py` — Embedding generation and vectorstore creation.
- `app.py` — (Optional) Streamlit user interface.
- `weather_faiss_vectorstore_main/` — FAISS vectorstore folder (should be ignored in `.gitignore`).

## Setup

1. **Clone the repository:**
   ```
   git clone https://github.com/<your-username>/<repo-name>.git
   cd RAG_Setup
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Download models:**
   - HuggingFace: `sentence-transformers/all-MiniLM-L6-v2`
   - Ollama: `qwen3:4b` (ensure Ollama server is running)

4. **Prepare vectorstore:**
   - Run `embed_gen.py` to generate FAISS index from your data.

5. **Run the main pipeline:**
   ```
   python RAG_main.py
   ```

## Usage

- Modify the query in `main(query)` to ask questions about your oceanographic dataset.
- The system retrieves relevant documents and generates concise, data-driven answers.

## Customization

- Update the prompt template in `RAG_main.py` for your specific data structure.
- Adjust FAISS search parameters (`k`) for more or fewer context documents.


Project Structure
Argo-RAG/
├── .gitignore              # Git ignore file
├── RAG_main.py             # Main RAG processing script
├── app.py                  # Streamlit dashboard script
├── argo_preprocessed_with_dates.csv  # Preprocessed Argo data with dates
├── embed_gen.py            # Script to generate embeddings

Usage

Dashboard: Access the Streamlit interface at http://localhost:8501 to visualize data and interact with queries.
RAG Processing: Run RAG_main.py to process data using generated embeddings.


Acknowledgements

Argo data from the Global Argo Data Repository.
