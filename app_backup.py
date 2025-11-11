import streamlit as st
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from main import research_topic
from qa_engine import make_qa_chain
from semantic_search import build_semantic_index, search_semantic
from pdf_utils import extract_text_from_pdf  # ✅ NEW IMPORT

# ---------- Page setup ----------
st.set_page_config(
    page_title="Athena – AI Research Assistant",
    page_icon="🧠",
    layout="wide"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
h1, h2, h3 {
    color: #1e3a8a;
    font-family: 'Segoe UI', sans-serif;
}
.stTextInput>div>div>input {
    border-radius: 8px;
    border: 1px solid #d1d5db;
    padding: 10px;
}
.stButton>button {
    background-color: #2563eb;
    color: white;
    font-weight: 500;
    border-radius: 8px;
    height: 3em;
    width: 12em;
    border: none;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #1e40af;
}
.result-box {
    background-color: white;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 0 10px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.title("🧠 Athena")
st.markdown("### _Local AI Research Assistant powered by Ollama_")
st.markdown("---")

# ---------- Input section ----------
st.markdown("### 📄 Upload a research paper (PDF) or enter a topic")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

topic = st.text_input(
    "🔍 Research Topic",
    placeholder="e.g. Recent advances in computer vision"
)

# ---------- Summarization ----------
if st.button("✨ Research", key="research_button"):
    if topic.strip() == "" and not uploaded_file:
        st.warning("Please enter a topic or upload a PDF.")
    else:
        with st.spinner("Collecting and analyzing data..."):
            # --- Case 1: Uploaded PDF ---
            if uploaded_file:
                try:
                    # Use improved PDF extraction
                    text = extract_text_from_pdf(uploaded_file)
                    
                    if not text.strip():
                        st.error("❌ Could not extract text from PDF. The file might be scanned or encrypted.")
                        st.stop()

                    # Save PDF text for Q&A and Search
                    st.session_state.pdf_text = text
                    st.session_state.pdf_uploaded = True

                    # Split into chunks
                    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
                    chunks = splitter.split_text(text)

                    summaries = []
                    st.info(f"Processing {len(chunks)} sections...")
                    progress_bar = st.progress(0)

                    for i, chunk in enumerate(chunks):
                        partial_query = f"Summarize this section of a research paper:\n\n{chunk}"
                        chunk_summary = research_topic(partial_query, skip_tools=True)
                        summaries.append(chunk_summary)
                        progress_bar.progress((i + 1) / len(chunks))

                    combined_text = "\n\n".join(summaries)
                    final_query = f"Combine the following section summaries into one cohesive academic summary:\n\n{combined_text}"
                    result = research_topic(final_query, skip_tools=True)

                except Exception as e:
                    st.error(f"❌ Error processing PDF: {e}")
                    st.stop()

            # --- Case 2: Topic-based research ---
            else:
                try:
                    result = research_topic(topic)
                    st.session_state.pdf_text = result
                    st.session_state.pdf_uploaded = True
                except Exception as e:
                    st.error(f"❌ Error during research: {e}")
                    st.stop()

        # Store result
        st.session_state.last_result = result

        # --- Tabs Layout ---
        tab1, tab2, tab3 = st.tabs(["📄 Summary", "💬 Q&A", "🔍 Semantic Search"])

        # ---------- Summary TAB ----------
        with tab1:
            st.markdown("### 📋 Research Summary")
            st.markdown(f"<div class='result-box'>{result}</div>", unsafe_allow_html=True)

            st.download_button(
                label="💾 Download Summary",
                data=result,
                file_name="athena_summary.txt",
                mime="text/plain",
                key="download_summary"
            )

        # ---------- Q&A TAB ----------
        with tab2:
            st.markdown("### 💬 Ask Questions about this Paper")

            if st.session_state.get("pdf_uploaded", False) and "pdf_text" in st.session_state:
                # Build Q&A index if not exists
                if "qa_chain" not in st.session_state:
                    st.info("⏳ Building Q&A index for the uploaded paper... Please wait.")
                    try:
                        with st.spinner("Generating embeddings and FAISS retriever..."):
                            qa_function = make_qa_chain(
                                st.session_state.pdf_text,
                                chunk_size=2000,
                                k=3,
                                model="llama3"
                            )
                            st.session_state.qa_chain = qa_function
                        st.success("✅ Q&A index ready! You can now ask questions.")
                    except Exception as e:
                        st.error(f"❌ Error building Q&A index: {e}")
                        st.stop()

                # Q&A Interface
                query = st.text_input("Your question:", placeholder="e.g. What dataset did this paper use?", key="qa_input")
                
                if st.button("Ask", key="ask_button"):
                    if query.strip() == "":
                        st.warning("Please enter a question.")
                    else:
                        with st.spinner("Thinking..."):
                            try:
                                # Call the Q&A function
                                answer = st.session_state.qa_chain(query)
                                st.markdown("**Answer:**")
                                st.markdown(f"<div class='result-box'>{answer}</div>", unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"❌ Error getting answer: {e}")
            else:
                st.warning("⚠️ Upload or summarize a PDF first to enable Q&A mode.")

        # ---------- SEMANTIC SEARCH TAB ----------
        with tab3:
            st.markdown("### 🔍 Semantic Search in Paper")

            if not st.session_state.get("pdf_uploaded", False) or "pdf_text" not in st.session_state:
                st.warning("⚠️ Please upload and extract a PDF first.")
            elif not st.session_state["pdf_text"].strip():
                st.warning("⚠️ PDF text is empty. Please upload a valid PDF.")
            else:
                # Build semantic index if not exists
                if "semantic_index" not in st.session_state:
                    with st.spinner("⚙️ Building semantic index..."):
                        try:
                            vectordb = build_semantic_index(st.session_state.pdf_text)
                            st.session_state.semantic_index = vectordb
                            st.success("✅ Semantic index built successfully!")
                        except Exception as e:
                            st.error(f"❌ Error building semantic index: {e}")
                            st.stop()

                # Search Interface
                query = st.text_input("Enter your semantic search query:", key="semantic_input", 
                                    placeholder="e.g. methodology used in the study")

                if st.button("Search", key="semantic_button"):
                    if not query.strip():
                        st.warning("Please enter a search query.")
                    else:
                        with st.spinner("🔍 Performing semantic search..."):
                            try:
                                results = search_semantic(st.session_state.semantic_index, query, k=5)
                                
                                if not results:
                                    st.warning("😶 No relevant matches found.")
                                else:
                                    st.session_state.semantic_results = results
                                    st.success(f"✅ Found {len(results)} relevant results.")
                            except Exception as e:
                                st.error(f"❌ Error during search: {e}")

                # Display persistent results
                if "semantic_results" in st.session_state and st.session_state.semantic_results:
                    st.markdown("### 🧩 Semantic Matches:")
                    for i, (text, score) in enumerate(st.session_state.semantic_results, 1):
                        st.markdown(f"**Match {i} (Similarity: {score:.4f})**")
                        st.markdown(f"<div class='result-box'>{text}</div>", unsafe_allow_html=True)
                        st.markdown("---")


# ---------- Footer ----------
st.markdown("---")
st.caption("Athena © 2025 · Built with Streamlit & LangChain")