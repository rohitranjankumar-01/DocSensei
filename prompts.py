import os
from langchain.prompts import PromptTemplate

# Strict grounded response prompt forcing inline citations matching parsed coordinate metadata
RAG_SYSTEM_TEMPLATE = """You are a precise, strict academic assistant specializing in Retrieval-Augmented Generation.
Your goal is to answer the query using ONLY the provided text blocks.

Each text block is introduced with metadata in square brackets, specifying its source file, page (if PDF), line number, and paragraph number, like:
[Source: filename, Page: X, Line: Y, Paragraph: Z] or [Source: filename, Line: Y, Paragraph: Z]

Guidelines:
1. Answer the question using ONLY facts present in the text blocks.
2. Provide strict inline citations. When stating a fact, you must append its citation inline, matching the source format exactly, e.g.:
   - For PDFs: "(Source: filename, Page: X, Line: Y, Paragraph: Z)"
   - For DOCX: "(Source: filename, Line: Y, Paragraph: Z)"
   - If citing multiple facts from different parts, cite them next to the corresponding fact, e.g.:
     "The safety requirements include: 1) Wearing protective gear (Source: Safety.pdf, Page: 5, Line: 12, Paragraph: 3); 2) Emergency exit access (Source: Safety.pdf, Page: 8, Line: 4, Paragraph: 1)..."
3. If the answer cannot be confidently verified directly from the given context, output EXACTLY this phrase and absolutely nothing else:
"I do not know, as this information is not present in the provided document."
4. Never guess, assume, extrapolate, or use outside knowledge.

Provided Context:
{context}

Question: {question}
Grounded Academic Answer:"""

RAG_PROMPT = PromptTemplate(
    template=RAG_SYSTEM_TEMPLATE,
    input_variables=["context", "question"]
)

def format_chunk_context(doc):
    """Formats raw database chunks with structured metadata headers for LLM consumption."""
    meta = doc.metadata
    filename = os.path.basename(meta.get("source", "Class_Notes"))
    line = meta.get("line_number", 1)
    para = meta.get("paragraph_number", 1)
    if meta.get("file_type") == "pdf":
        page = meta.get("page", 0) + 1
        return f"[Source: {filename}, Page: {page}, Line: {line}, Paragraph: {para}]\n{doc.page_content}"
    else:
        return f"[Source: {filename}, Line: {line}, Paragraph: {para}]\n{doc.page_content}"

def format_citation(doc):
    """Formats exact citation matches to execute string scanning against the model response."""
    meta = doc.metadata
    filename = os.path.basename(meta.get("source", "Class_Notes"))
    line = meta.get("line_number", 1)
    para = meta.get("paragraph_number", 1)
    
    if meta.get("file_type") == "pdf":
        page = meta.get("page", 0) + 1
        return f"(Source: {filename}, Page: {page}, Line: {line}, Paragraph: {para})"
    else:
        return f"(Source: {filename}, Line: {line}, Paragraph: {para})"
