import warnings

# Suppress noisy runtime, cryptography, and deprecation warnings in terminal console logs
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
# Disable ChromaDB anonymized telemetry globally
os.environ["ANONYMIZED_TELEMETRY"] = "False"



import streamlit as st
from dotenv import load_dotenv
import config
import ingestion
import vectorstore
import llm_providers
import prompts



# Main File Ingestion Uploader
uploaded_file = st.file_uploader("Upload Class Lecture Notes (.pdf, .docx)", type=["pdf", "docx"])

if uploaded_file:
    # Trigger ingestion and indexing pipeline if the uploaded file is new
    if st.session_state.indexed_backends.get(backend) != uploaded_file.name:
        vectorstore.reset_collections(backend, embeddings)
        st.session_state.messages = []
        
        temp_dir = "temp_processing_store"
        os.makedirs(temp_dir, exist_ok=True)
        target_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(target_path, "wb") as buffer:
            buffer.write(uploaded_file.getbuffer())
            
        with st.spinner(f"Analyzing and indexing document for {backend}..."):
            try:
                chunks = ingestion.process_document(target_path)
                v_store = vectorstore.get_vector_store(backend, embeddings)
                v_store.add_documents(chunks)
                st.session_state.indexed_backends[backend] = uploaded_file.name
                st.success(f"Successfully processed and indexed document blocks for {backend}!")
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "quota" in err_msg.lower():
                    st.error("**Google Gemini API Quota Exceeded during Indexing:** You have hit your free-tier embedding rate limits. Please wait a minute and re-upload the file to retry.")
                else:
                    st.error(f"Indexing error: {err_msg}")
                st.session_state.indexed_backends[backend] = None
                st.stop()
            finally:
                if os.path.exists(target_path):
                    os.remove(target_path)

    # Render previous conversation history from session state
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "citations" in msg and msg["citations"]:
                st.markdown("<br><strong>Cited References:</strong><br>", unsafe_allow_html=True)
                for cit in msg["citations"]:
                    st.markdown(f'<span class="citation-badge">{cit}</span>', unsafe_allow_html=True)

    # User query chat interface
    user_query = st.chat_input("Ask a question regarding your notes...")
    
    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Retrieve context chunks matching user query
        v_store = vectorstore.get_vector_store(backend, embeddings)
        matched_results = v_store.similarity_search_with_score(user_query, k=config.RETRIEVAL_K)
        
        if not matched_results:
            response_text = "I do not know, as this information is not present in the provided document."
            with st.chat_message("assistant"):
                st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        else:
            # Distance score evaluation matching config settings
            _, primary_score = matched_results[0]
            configured_cutoff = config.THRESHOLD_API if backend == "API Mode" else config.THRESHOLD_LOCAL
            
            if debug_mode:
                st.info(f"Debug Tracking Vector Score: {primary_score:.4f} (Active Threshold Limit: {configured_cutoff})")
                
            # If closest vector distance is above configured threshold, query falls outside document context
            if primary_score > configured_cutoff:
                response_text = "I do not know, as this information is not present in the provided document."
                with st.chat_message("assistant"):
                    st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            else:
                # Compile retrieved context blocks and formulate final prompt
                context_string = "\n\n".join([prompts.format_chunk_context(c[0]) for c in matched_results])
                formatted_prompt = prompts.RAG_PROMPT.format(context=context_string, question=user_query)
                
                with st.chat_message("assistant"):
                    with st.spinner("Formulating grounded response..."):
                        try:
                            raw_llm_response = llm.invoke(formatted_prompt)
                            final_text = getattr(raw_llm_response, 'content', str(raw_llm_response)).strip()
                            
                            st.markdown(final_text)
                            
                            # Filter and badge references that are actually present inside the response text
                            citations = []
                            if "I do not know, as this information is not present" not in final_text:
                                normalized_response = " ".join(final_text.lower().split())
                                for c in matched_results:
                                    cit_str = prompts.format_citation(c[0])
                                    normalized_cit = " ".join(cit_str.lower().split())
                                    # If model referenced this coordinate, add it to citations list
                                    if normalized_cit in normalized_response:
                                        citations.append(cit_str)
                                
                                # Remove duplicate citations preserving retrieval order
                                seen = set()
                                citations = [x for x in citations if not (x in seen or seen.add(x))]
                                
                                if citations:
                                    st.markdown("<br><strong>Cited References:</strong><br>", unsafe_allow_html=True)
                                    for cit in citations:
                                        display_badge = cit.strip("()")
                                        st.markdown(f'<span class="citation-badge">{display_badge}</span>', unsafe_allow_html=True)
                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": final_text,
                                "citations": [cit.strip("()") for cit in citations]
                            })
                        except Exception as api_err:
                            err_msg = str(api_err)
                            if "429" in err_msg or "quota" in err_msg.lower():
                                st.error("**Google Gemini API Quota Exceeded:** You have hit your free-tier rate limits or daily requests quota. Please check your plan details, or wait a minute before retrying.")
                            else:
                                st.error(f"Execution exception: {err_msg}")
