import os

# --- Path Configurations ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_LOCAL_DIR = os.path.join(BASE_DIR, "chroma_db", "local")
CHROMA_API_DIR = os.path.join(BASE_DIR, "chroma_db", "api")

# --- Text Splitting Configurations ---
# Reduced chunk size ensures high line/paragraph alignment precision
CHUNK_SIZE = 250
CHUNK_OVERLAP = 30

# --- Retrieval Settings ---
RETRIEVAL_K = 4 

# --- Distance Threshold Cutoffs (L2 Bounded Distance) ---
THRESHOLD_LOCAL = 1.10  
THRESHOLD_API = 0.75    

# --- Active Model Name Specifications ---
LOCAL_LLM = "doc-sensei-llama3.1"
LOCAL_EMBED = "doc-sensei-nomic-embed-text"
API_LLM = "gemini-3.1-flash-lite"
API_EMBED = "models/gemini-embedding-001"