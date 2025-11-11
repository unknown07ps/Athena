import streamlit as st
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from main import research_topic
from qa_engine import make_qa_chain
from semantic_search import build_semantic_index, search_semantic
from pdf_utils import extract_text_from_pdf
from chat_engine import AthenaChat
from document_comparison import DocumentComparison

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
    line-height: 1.7;
}
.answer-box {
    background-color: #eff6ff;
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #3b82f6;
    line-height: 1.7;
}
.comparison-box {
    background-color: #f0fdf4;
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #22c55e;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.title("🧠 Athena")
st.markdown("### _Local AI Research Assistant powered by Ollama_")
st.markdown("---")

# ---------- Initialize Session State ----------
if "athena_chat" not in st.session_state:
    st.session_state.athena_chat = AthenaChat(model="llama3", temperature=0.3)

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "doc_comparison" not in st.session_state:
    st.session_state.doc_comparison = DocumentComparison(model="llama3")

if "comparison_docs" not in st.session_state:
    st.session_state.comparison_docs = []

# ---------- Input section ----------
st.markdown("### 📄 Upload a research paper (PDF) or enter a topic")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

topic = st.text_input(
    "🔎 Research Topic",
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
                    text = extract_text_from_pdf(uploaded_file)
                    
                    if not text.strip():
                        st.error("❌ Could not extract text from PDF. The file might be scanned or encrypted.")
                        st.stop()

                    st.session_state.pdf_text = text
                    st.session_state.pdf_uploaded = True

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
        
        # Set PDF context for chat engine
        st.session_state.athena_chat.set_pdf_context(st.session_state.pdf_text)
        
        st.success("✅ Research complete! Check the tabs below.")

# ---------- Show Tabs OUTSIDE the button (persistent) ----------
if st.session_state.pdf_uploaded and st.session_state.last_result:
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Summary", 
        "💬 Q&A", 
        "🔍 Semantic Search",
        "🤖 Chat with Athena",
        "📊 Document Comparison"
    ])

    # ---------- Summary TAB ----------
    with tab1:
        st.markdown("### 📋 Research Summary")
        st.markdown(f"<div class='result-box'>{st.session_state.last_result}</div>", unsafe_allow_html=True)

        st.download_button(
            label="💾 Download Summary",
            data=st.session_state.last_result,
            file_name="athena_summary.txt",
            mime="text/plain",
            key="download_summary"
        )

    # ---------- Q&A TAB ----------
    with tab2:
        st.markdown("### 💬 Ask Questions about this Paper")

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

        query = st.text_input("Your question:", placeholder="e.g. What dataset did this paper use?", key="qa_input")
        
        if st.button("Ask", key="ask_button"):
            if query.strip() == "":
                st.warning("Please enter a question.")
            else:
                with st.spinner("Thinking..."):
                    try:
                        answer = st.session_state.qa_chain(query)
                        st.markdown("**Answer:**")
                        st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"❌ Error getting answer: {e}")

    # ---------- SEMANTIC SEARCH TAB ----------
    with tab3:
        st.markdown("### 🔍 Semantic Search in Paper")

        if "semantic_index" not in st.session_state:
            with st.spinner("⚙️ Building semantic index..."):
                try:
                    vectordb = build_semantic_index(st.session_state.pdf_text, 
                                                    chunk_size=300, 
                                                    chunk_overlap=50)
                    st.session_state.semantic_index = vectordb
                    st.success("✅ Semantic index built successfully!")
                except Exception as e:
                    st.error(f"❌ Error building semantic index: {e}")
                    st.stop()

        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "Enter your semantic search query:", 
                key="semantic_input", 
                placeholder="e.g. methodology used in the study"
            )
        
        with col2:
            num_results = st.selectbox("Results", [5, 10, 15, 20], index=0, key="num_results")

        if st.button("Search", key="semantic_button"):
            if not query.strip():
                st.warning("Please enter a search query.")
            else:
                with st.spinner("🔍 Performing semantic search..."):
                    try:
                        results = search_semantic(
                            st.session_state.semantic_index, 
                            query, 
                            k=num_results
                        )
                        
                        if not results:
                            st.warning("😶 No relevant matches found.")
                        else:
                            st.session_state.semantic_results = results
                            st.success(f"✅ Found {len(results)} relevant results.")
                    except Exception as e:
                        st.error(f"❌ Error during search: {e}")

        if "semantic_results" in st.session_state and st.session_state.semantic_results:
            st.markdown("### 🧩 Semantic Matches:")
            
            min_similarity = st.slider(
                "Minimum Similarity Score", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.3, 
                step=0.05,
                help="Filter results by minimum similarity (higher = more relevant)"
            )
            
            filtered_results = [
                (text, score) for text, score in st.session_state.semantic_results 
                if score >= min_similarity
            ]
            
            if not filtered_results:
                st.warning(f"No results above similarity threshold {min_similarity:.2f}")
            else:
                st.info(f"Showing {len(filtered_results)} results above {min_similarity:.2f} similarity")
                
                for i, (text, similarity) in enumerate(filtered_results, 1):
                    if similarity >= 0.7:
                        color = "🟢"
                    elif similarity >= 0.5:
                        color = "🟡"
                    else:
                        color = "🟠"
                    
                    with st.expander(f"{color} **Match {i}** - Similarity: {similarity:.2%}", expanded=(i<=3)):
                        st.markdown(f"```\n{text}\n```")

    # ---------- CHAT TAB ----------
    with tab4:
        st.markdown("### 🤖 Chat with Athena")
        st.info("💡 Athena has access to your uploaded document and will answer based on its content!")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col2:
            if st.button("🗑️ Clear Chat", key="clear_chat"):
                st.session_state.athena_chat.clear_history()
                st.session_state.chat_messages = []
                st.success("Chat cleared!")
                st.rerun()
        
        with col3:
            if st.button("💾 Export Chat", key="export_chat"):
                if st.session_state.chat_messages:
                    filename = st.session_state.athena_chat.export_history()
                    with open(filename, 'r', encoding='utf-8') as f:
                        chat_text = f.read()
                    
                    st.download_button(
                        label="📥 Download",
                        data=chat_text,
                        file_name=filename,
                        mime="text/plain"
                    )
                else:
                    st.warning("No chat history to export")
        
        st.markdown("---")
        
        if not st.session_state.chat_messages:
            st.info("👋 Start a conversation! Ask me anything about research, papers, or AI.")
        else:
            for msg in st.session_state.chat_messages:
                with st.chat_message("user"):
                    st.markdown(msg["user"])
                
                with st.chat_message("assistant", avatar="🧠"):
                    st.markdown(msg["assistant"])
        
        st.markdown("---")
        user_input = st.text_input(
            "Your message:",
            key="chat_input",
            placeholder="e.g., Explain the attention mechanism in transformers"
        )
        
        col1, col2 = st.columns([1, 5])
        
        with col1:
            send_button = st.button("Send 📤", key="send_chat")
        
        if send_button and user_input.strip():
            with st.spinner("Athena is thinking..."):
                response = st.session_state.athena_chat.chat(user_input)
                
                st.session_state.chat_messages.append({
                    "user": user_input,
                    "assistant": response
                })
                
                st.rerun()
        elif send_button:
            st.warning("Please enter a message")
        
        if st.session_state.chat_messages:
            st.markdown("---")
            st.caption(f"💬 {len(st.session_state.chat_messages)} exchanges in this conversation")

    # ---------- DOCUMENT COMPARISON TAB ----------
    with tab5:
        st.markdown("### 📊 Document Comparison")
        st.info("💡 Upload multiple PDFs to compare them side-by-side and find similarities/differences!")
        
        st.markdown("#### 📎 Upload Documents to Compare")
        
        col1, col2 = st.columns(2)
        
        with col1:
            doc1_file = st.file_uploader("Document 1", type="pdf", key="doc1_upload")
        
        with col2:
            doc2_file = st.file_uploader("Document 2", type="pdf", key="doc2_upload")
        
        if st.button("🔍 Compare Documents", key="compare_button"):
            if not doc1_file or not doc2_file:
                st.warning("⚠️ Please upload both documents to compare.")
            else:
                with st.spinner("📖 Extracting and analyzing documents..."):
                    try:
                        # Extract text from both PDFs
                        text1 = extract_text_from_pdf(doc1_file)
                        text2 = extract_text_from_pdf(doc2_file)
                        
                        if not text1.strip() or not text2.strip():
                            st.error("❌ Could not extract text from one or both PDFs.")
                            st.stop()
                        
                        # Add documents to comparison engine
                        st.session_state.doc_comparison.add_document(doc1_file.name, text1)
                        st.session_state.doc_comparison.add_document(doc2_file.name, text2)
                        
                        st.success("✅ Documents loaded successfully!")
                        
                        # Perform comparison
                        with st.spinner("🔬 Comparing documents..."):
                            comparison_result = st.session_state.doc_comparison.compare_documents(
                                doc1_file.name,
                                doc2_file.name
                            )
                            
                            st.session_state.comparison_result = comparison_result
                        
                    except Exception as e:
                        st.error(f"❌ Error during comparison: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        # Display comparison results
        if "comparison_result" in st.session_state and st.session_state.comparison_result:
            st.markdown("---")
            st.markdown("### 📈 Comparison Results")
            
            result = st.session_state.comparison_result
            
            # Summary
            st.markdown("#### 📝 Summary")
            st.markdown(f"<div class='comparison-box'>{result['summary']}</div>", unsafe_allow_html=True)
            
            # Similarities
            st.markdown("#### 🤝 Similarities")
            with st.expander("View Common Topics", expanded=True):
                st.markdown(f"<div class='result-box'>{result['similarities']}</div>", unsafe_allow_html=True)
            
            # Differences
            st.markdown("#### ⚡ Key Differences")
            with st.expander("View Unique Aspects", expanded=True):
                st.markdown(f"<div class='result-box'>{result['differences']}</div>", unsafe_allow_html=True)
            
            # Recommendations
            if result.get('recommendations'):
                st.markdown("#### 💡 Recommendations")
                with st.expander("View Insights"):
                    st.markdown(f"<div class='result-box'>{result['recommendations']}</div>", unsafe_allow_html=True)
            
            # Download comparison report
            st.markdown("---")
            report = f"""DOCUMENT COMPARISON REPORT
Generated by Athena AI Research Assistant

Document 1: {doc1_file.name if doc1_file else 'N/A'}
Document 2: {doc2_file.name if doc2_file else 'N/A'}

SUMMARY
{result['summary']}

SIMILARITIES
{result['similarities']}

DIFFERENCES
{result['differences']}

RECOMMENDATIONS
{result.get('recommendations', 'N/A')}
"""
            
            st.download_button(
                label="📥 Download Comparison Report",
                data=report,
                file_name="document_comparison_report.txt",
                mime="text/plain",
                key="download_comparison"
            )

# ---------- Footer ----------
st.markdown("---")
st.caption("Athena © 2025 · Built with Streamlit & LangChain")