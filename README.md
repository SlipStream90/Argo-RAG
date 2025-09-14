Argo-RAG
Overview
Argo-RAG is a lightweight Retrieval-Augmented Generation (RAG) system for processing and querying oceanographic data from Argo floats. It uses preprocessed Argo data with date annotations to enable semantic search via embeddings, with an interactive Streamlit dashboard.
Features

Data Preprocessing: Processes Argo float data (e.g., temperature, salinity) with dates.
Embedding Generation: Creates vector embeddings for semantic search.
RAG Implementation: Main script for RAG functionality.
Interactive Dashboard: Streamlit-based interface for visualizing and querying data.

Requirements

Python 3.10+
Dependencies: See requirements.txt (to be created if needed)
Minimum 4GB RAM

Setup Instructions

Clone the Repository:git clone https://github.com/SlipStream90/Argo-RAG.git
cd Argo-RAG


Set Up Python Environment:python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt  # Include Streamlit and other libraries


Run Embedding Generation:python embed_gen.py


Launch Streamlit Dashboard:streamlit run app.py


Run Main RAG Script:python RAG_main.py



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
