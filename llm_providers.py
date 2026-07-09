import streamlit as st
import subprocess
import os
from google import genai
from google.genai import types
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
import config
import numpy as np

class GoogleGenAIEmbeddingsWrapper:
    """Wraps the new Google GenAI SDK client to look like LangChain's Embeddings helper."""
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name

    def embed_documents(self, texts):
        # Wrap each text in types.Content to ensure individual processing across all embedding models
        contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in texts]
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=contents
        )
        return [emb.values for emb in response.embeddings]

    def embed_query(self, text):
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=text
        )
        return response.embeddings[0].values

class GoogleGenAIResponseWrapper:
    """Simple wrapper to emulate a LangChain chat response object with a .content attribute."""
    def __init__(self, content):
        self.content = content

class GoogleGenAILLMWrapper:
    """Wraps the new Google GenAI SDK client to look like a LangChain Chat Model."""
    def __init__(self, client, model_name, temperature=0.1):
        self.client = client
        self.model_name = model_name
        self.temperature = temperature

    def invoke(self, prompt):
        config_obj = types.GenerateContentConfig(
            temperature=self.temperature
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config_obj
        )
        return GoogleGenAIResponseWrapper(response.text)

class NormalizedEmbeddings:
    """Wraps an embedding function to yield L2-normalized vectors so that Euclidean distance corresponds directly to Cosine similarity (bounded 0 to 2)."""
    def __init__(self, base_embeddings):
        self.base_embeddings = base_embeddings
        
    def _normalize(self, vector):
        arr = np.array(vector)
        norm = np.linalg.norm(arr)
        if norm == 0:
            return vector
        return (arr / norm).tolist()
        
    def embed_documents(self, texts):
        vectors = self.base_embeddings.embed_documents(texts)
        return [self._normalize(v) for v in vectors]
        
    def embed_query(self, text):
        vector = self.base_embeddings.embed_query(text)
        return self._normalize(vector)

@st.cache_resource
def load_api_components(api_key):
    """Instantiates and caches cloud-based embedding and generation models."""
    client = genai.Client(api_key=api_key)
    base_embeddings = GoogleGenAIEmbeddingsWrapper(client, config.API_EMBED)
    embeddings = NormalizedEmbeddings(base_embeddings)
    llm = GoogleGenAILLMWrapper(client, config.API_LLM, temperature=0.1)
    return embeddings, llm

def ensure_custom_ollama_model():
    """Checks if the custom GGUF-based local model is registered in Ollama
    and if the required local embedding model is pulled. If not, registers/pulls them.
    """
    try:
        # Run 'ollama list' to check existing registered models
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        
        # 1. Handle Custom LLM Creation
        model_name = config.LOCAL_LLM
        if model_name == "doc-sensei-llama3.1" and model_name not in result.stdout:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            modelfile_path = os.path.join(base_dir, "llama_model", "Modelfile")
            if os.path.exists(modelfile_path):
                subprocess.run(["ollama", "create", model_name, "-f", modelfile_path], check=True)
                
        # 2. Handle Embedding Model Setup
        embed_name = config.LOCAL_EMBED
        if embed_name not in result.stdout:
            if embed_name == "doc-sensei-nomic-embed-text":
                base_dir = os.path.dirname(os.path.abspath(__file__))
                modelfile_path = os.path.join(base_dir, "llama_model", "Modelfile_embed")
                if os.path.exists(modelfile_path):
                    subprocess.run(["ollama", "create", embed_name, "-f", modelfile_path], check=True)
            else:
                subprocess.run(["ollama", "pull", embed_name], check=True)
            
    except Exception:
        # Don't block application startup if Ollama CLI is not on PATH or fails;
        # let the subsequent connection check handle the direct error.
        pass

@st.cache_resource
def load_local_components():
    """Instantiates and caches local embedding and generation models."""
    # Ensure custom local GGUF model is registered in Ollama
    ensure_custom_ollama_model()
    try:
        base_embeddings = OllamaEmbeddings(model=config.LOCAL_EMBED)
        embeddings = NormalizedEmbeddings(base_embeddings)
        llm = Ollama(model=config.LOCAL_LLM, temperature=0.1)
        return embeddings, llm
    except Exception as e:
        raise RuntimeError("Ollama server connection refused. Ensure Ollama app is locally running and the custom model is created.") from e