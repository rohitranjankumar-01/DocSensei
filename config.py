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