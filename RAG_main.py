from transformers import pipeline,AutoTokenizer,BitsAndBytesConfig,AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.llms import ollama
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_ollama.llms import OllamaLLM
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings import OllamaEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate

model_name_1 = "qwen3:4b"

#hf_pipe=pipeline("text-generation",model=model_main,tokenizer=tokenizer,temperature=0.1,top_p=0.75,max_new_tokens=32000)
#llm = HuggingFacePipeline(pipeline=hf_pipe)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
try:
    # This will download and cache the model
    temp_model = SentenceTransformer(model_name)
    print("Model downloaded successfully!")
except:
    print("Model download failed, trying alternative...")
    model_name = "all-MiniLM-L6-v2"

model_kwargs = {'device': 'cpu'}  # Use 'cuda' if GPU available
encode_kwargs = {'normalize_embeddings': False}
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vectorstore = FAISS.load_local(folder_path="weather_faiss_vectorstore_main",allow_dangerous_deserialization=True,embeddings=hf_embeddings)

retriever=vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":3})  # Increased k for better context

llm=OllamaLLM(model=model_name_1,base_url="http://localhost:11434",num_predict=2048,temperature=0.1,top_p=0.75)

# Create custom prompt template for oceanographic data
custom_prompt_template = """You are an expert oceanographer analyzing marine data. Use the following oceanographic data to answer the question.

The data contains measurements with this structure:
- Numbers represent: [ID] [Depth] [Pressure] [Temperature] [Salinity] [Station_ID] [Other] [Latitude] [Longitude] [Timestamp] [Date]
- Temperature is in degrees Celsius
- Salinity is in practical salinity units (PSU)  
- Depth/Pressure measurements in meters/decibars
- Coordinates are in decimal degrees (negative values indicate South/West)
- Dates are in YYYY-MM-DD format

Context Data:
{context}

Question: {question}

When answering:
- Look for exact date first, then within ±7 days if needed
- If using nearby date data, mention the actual date and day difference
- Provide specific measurements with units
- Include location coordinates
- Be direct and concise
- Do not repeat these instructions in your response
- Focus only on the data and findings

Answer:"""

# Create the prompt template
PROMPT = PromptTemplate(
    template=custom_prompt_template,
    input_variables=["context", "question"]
)

# Create QA chain with custom prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

def run_query(query):
    result = qa_chain.invoke({"query": query})
    return result['result'], len(result['source_documents']), result['source_documents']

def show_retrieved_docs(docs):
    """Helper function to display retrieved documents"""
    print("\n" + "="*60)
    print("RETRIEVED DOCUMENTS:")
    print("="*60)
    for i, doc in enumerate(docs, 1):
        print(f"\nDocument {i}:")
        print(f"Metadata: {doc.metadata}")
        print(f"Content: {doc.page_content}")
        print("-" * 40)

def main(query):
    answer, num_docs, source_docs = run_query(query)
    return answer, num_docs