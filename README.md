# DocSensei: Production-Grade Grounded RAG Application

DocSensei is a production-grade, modular Retrieval-Augmented Generation (RAG) platform optimized for processing educational curriculum materials and enterprise documents without data leakages or citation hallucinations.

---

## 🗺️ System Architecture

The following diagram illustrates the data flow of a user query through the DocSensei application:

```mermaid
graph TD
    User[User] -->|Asks Question / Uploads Doc| UI[Streamlit UI]
    UI -->|Loads Config & Active Mode| Selector[Model Selector]
    Selector -->|Constructs Prompts & Query| LangChain[LangChain Pipeline]
    LangChain -->|Retrieves Chunks with Scores| ChromaDB[ChromaDB Vector Store]
    ChromaDB -->|Returns Context Chunks| LangChain
    LangChain -->|Passes Prompt + Guardrails| LLM[LLM]
    LLM -->|Synthesizes Grounded Answer| Output[Output UI Console]
```

---

## ✨ Core Features

1. **"Model Switcher" Engine:** A sidebar toggle allows swapping runtime components:
   * **Local Mode:** Powered completely offline by `Ollama` using `llama3.1` and `nomic-embed-text` embeddings.
   * **API Mode:** Powered by the `Google Gemini API` using `gemini-embedding-001` for vector embedding. It includes a sub-model switcher to dynamically swap the generation LLM (supporting *Gemini 3.1 Flash Lite*, *Gemini 3.5 Flash*, *Gemini 2.5 Flash Lite*, and *Gemini 3 Flash*) without affecting or requiring re-indexing of the document vector store.
2. **Grounded Retrieval (Anti-Hallucination Guardrails):**
   * Multi-stage safety checks prevent hallucinations.
   * A mathematical vector distance pre-flight check rejects questions whose closest matches are beyond a distance cutoff threshold.
   * If the LLM determines context is missing, it is constrained to output exactly: *"I do not know, as this information is not present in the provided document."* (Refusals suppress cited page badges).
3. **Modularity:** Separation of concerns ensures data loaders (`ingestion.py`), database structures (`vectorstore.py`), and model layers (`llm_providers.py`) are fully independent.

---

## 📂 File Directory Layout

```
├── llama_model/
│   ├── Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
│   ├── Modelfile
│   ├── Modelfile_embed
│   └── nomic-embed-text.gguf
├── README.md
├── app.py
├── config.py
├── ingestion.py
├── llm_providers.py
├── prompts.py
├── requirements.txt
├── runtime.txt
├── setup_n_launch.bat
└── vectorstore.py
```

*   `app.py`: Streamlit frontend dashboard, custom styling injection, and pipeline execution orchestration.
*   `config.py`: Threshold cutoffs, chunk sizes, path configurations, and model definitions.
*   `ingestion.py`: PyPDF and DOCX document processing loaders and recursive chunking logic.
*   `llm_providers.py`: Instantiation of API (Gemini) and Local (Ollama/llama.cpp) LLM engines.
*   `prompts.py`: Strict instruction directives for citation grounding and answer formatting.
*   `requirements.txt`: Python package dependencies.
*   `runtime.txt`: Version declaration of the Python runtime.
*   `setup_n_launch.bat`: Automated environment build and dashboard startup batch script.
*   `vectorstore.py`: Local ChromaDB vector database creation, querying, and maintenance script.
*   `llama_model/`: Local model repository directory.
    *   `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`: Quantized local Llama 3.1 8B Instruct model.
    *   `nomic-embed-text.gguf`: Local text embedding model.
    *   `Modelfile` / `Modelfile_embed`: Manifest configuration files to build/import the model formats in Ollama.

---

## 🚀 Quick Start Guide

You can run DocSensei in either **API Mode (Cloud)** or **Local Mode (Offline)**. Follow the steps below to configure your environment and launch the application.

### 📋 Prerequisites

1. **Python 3.11:** Ensure Python 3.11 is installed on your system. The launch script specifically runs `py -3.11` to build the environment.
2. **Active Internet Connection:** Required for the first run to build the virtual environment and install dependencies.

---

### ⚙️ Configuration

#### Option A: API Mode (Google Gemini API) — *Recommended for instant setup*
1. Obtain an API key from [Google AI Studio](https://aistudio.google.com/).
2. Create a `.env` file in the root directory of this project.
3. Define your API key:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

#### Option B: Local Mode (Ollama) — *Completely offline*
1. Install [Ollama](https://ollama.com/) on your local machine.
2. Ensure the Ollama background service is running (accessible at `http://localhost:11434`).
3. Place the following quantized model files inside the `llama_model/` directory:
   * `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` (Local LLM)
   * `nomic-embed-text.gguf` (Local Embeddings)
4. The application will automatically register these custom models (`doc-sensei-llama3.1` and `doc-sensei-nomic-embed-text`) inside Ollama upon switching to Local Mode.

---

### 🏎️ Launching the Application

1. Double-click the `setup_n_launch.bat` batch script in the root directory (or execute it from a terminal).
2. The batch script will automatically:
   * Create a Python 3.11 virtual environment (`venv`).
   * Upgrade packaging utilities (`pip`, `setuptools`, `wheel`).
   * Install all requirements listed in `requirements.txt`.
   * Start the Streamlit application interface.
3. Streamlit will launch and open your default browser to `http://localhost:8501`.
