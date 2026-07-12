import streamlit as st
import subprocess
import os
import requests
from google import genai
from google.genai import types
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
import config
import numpy as np

class BatchedOllamaEmbeddings:
    """Custom wrapper to call Ollama's batched /api/embed API for 30x+ faster indexing."""
    def __init__(self, model, base_url="http://localhost:11434", num_gpu=99):
        self.model = model
        self.base_url = base_url
        self.num_gpu = num_gpu

    def _embed(self, texts):
        batch_size = 128
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                res = requests.post(
                    f"{self.base_url}/api/embed",
                    json={
                        "model": self.model,
                        "input": batch,
                        "options": {
                            "num_gpu": self.num_gpu
                        }
                    },
                    headers={"Content-Type": "application/json"}
                )
                res.raise_for_status()
                data = res.json()
                all_embeddings.extend(data["embeddings"])
            except Exception as e:
                raise ValueError(f"Error calling Ollama batch embed: {e}")
        return all_embeddings

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_query(self, text):
        return self._embed([text])[0]

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
def load_api_components(api_key, llm_model_name=None):
    """Instantiates and caches cloud-based embedding and generation models."""
    if llm_model_name is None:
        llm_model_name = config.API_LLM
    client = genai.Client(api_key=api_key)
    base_embeddings = GoogleGenAIEmbeddingsWrapper(client, config.API_EMBED)
    embeddings = NormalizedEmbeddings(base_embeddings)
    llm = GoogleGenAILLMWrapper(client, llm_model_name, temperature=0.1)
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
    
    # Actively verify Ollama server connection and check for required models
    try:
        res = requests.get("http://localhost:11434/api/tags", timeout=5)
        res.raise_for_status()
        tags_data = res.json()
        model_names = [m["name"] for m in tags_data.get("models", [])]
        
        # Check if local generation and embedding models are present (supporting optional ':latest' tag suffix)
        for target_model in [config.LOCAL_LLM, config.LOCAL_EMBED]:
            found = False
            for name in model_names:
                if name == target_model or name.startswith(f"{target_model}:"):
                    found = True
                    break
            if not found:
                raise ValueError(f"Required model '{target_model}' is not registered in Ollama. Please run setup_n_launch.bat or register/pull it manually.")
                
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError("Ollama server connection refused. Ensure Ollama app is running locally on port 11434.") from e
    except Exception as e:
        raise RuntimeError(f"Ollama local validation failed: {e}") from e

    try:
        base_embeddings = BatchedOllamaEmbeddings(model=config.LOCAL_EMBED, num_gpu=99)
        embeddings = NormalizedEmbeddings(base_embeddings)
        llm = Ollama(model=config.LOCAL_LLM, temperature=0.1, num_gpu=99, num_ctx=2048)
        return embeddings, llm
    except Exception as e:
        raise RuntimeError("Ollama server connection refused. Ensure Ollama app is locally running and the custom model is created.") from e