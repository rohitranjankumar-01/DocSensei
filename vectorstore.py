import os
import shutil
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
import config

def get_vector_store(backend_mode, embedding_function):
    """Retrieves or builds the Chroma vector database collection matching the current active backend."""
    persist_dir = config.CHROMA_API_DIR if backend_mode == "API Mode" else config.CHROMA_LOCAL_DIR
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_function,
        client_settings=Settings(anonymized_telemetry=False)
    )

def reset_collections(backend_mode, embedding_function):
    """Clears existing vector collections, with a fallback to rmtree directory deletion if database locks occur."""
    try:
        store = get_vector_store(backend_mode, embedding_function)
        store.delete_collection()
    except Exception:
        # Fallback directory deletion
        persist_dir = config.CHROMA_API_DIR if backend_mode == "API Mode" else config.CHROMA_LOCAL_DIR
        if os.path.exists(persist_dir):
            try:
                shutil.rmtree(persist_dir)
            except Exception:
                pass