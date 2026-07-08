import os
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docx import Document as DocxReader
from langchain_core.documents import Document
import config

def load_docx_safely(file_path):
    """Loads a DOCX document into a single text block without fabricating fake page numbers."""
    doc = DocxReader(file_path)
    full_text = [p.text for p in doc.paragraphs if p.text.strip()]
    return [Document(
        page_content="\n".join(full_text), 
        metadata={"source": os.path.basename(file_path), "file_type": "docx"}
    )]

def process_document(file_path):
    """Parses PDF/DOCX documents, splits them into chunks, and computes line and paragraph offsets."""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # 1. Load document pages/data
    if file_ext == ".pdf":
        loader = PyPDFLoader(file_path)
        raw_docs = loader.load()
        for d in raw_docs:
            d.metadata["source"] = os.path.basename(d.metadata["source"])
            d.metadata["file_type"] = "pdf"
    elif file_ext in [".docx", ".doc"]:
        raw_docs = load_docx_safely(file_path)
    else:
        raise ValueError("Unsupported extension profile encountered.")
        
    # 2. Split document text into granular chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True
    )
    chunks = splitter.split_documents(raw_docs)
    
    # 3. Calculate exact line and paragraph coordinates for each chunk
    for c in chunks:
        parent_doc = None
        if file_ext == ".pdf":
            page_num = c.metadata.get("page")
            for rd in raw_docs:
                if rd.metadata.get("page") == page_num:
                    parent_doc = rd
                    break
        else:
            parent_doc = raw_docs[0]
            
        if parent_doc:
            start_idx = c.metadata.get("start_index", 0)
            before_text = parent_doc.page_content[:start_idx]
            # Count standard newlines and double newlines to find line/para offsets
            c.metadata["line_number"] = before_text.count("\n") + 1
            c.metadata["paragraph_number"] = before_text.count("\n\n") + 1
        else:
            c.metadata["line_number"] = 1
            c.metadata["paragraph_number"] = 1
            
    return chunks