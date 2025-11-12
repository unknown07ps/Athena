import streamlit as st
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from main import research_topic
from qa_engine import make_qa_chain
from semantic_search import build_semantic_index, search_semantic
from pdf_utils import extract_text_from_pdf
from chat_engine import AthenaChat

# Try to import optional features
try:
    from document_comparison import DocumentComparison
    COMPARISON_AVAILABLE = True
except ImportError:
    COMPARISON_AVAILABLE = False
    print("⚠️ Document comparison not available - document_comparison.py missing")

try:
    from voice_interface import render_voice_tab
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    print("⚠️ Voice interface not available - voice_interface.py missing")

try:
    from kg_visualizer import render_knowledge_graph_tab
    from advanced_rag import AdvancedRAG
    KG_RAG_AVAILABLE = True
except ImportError:
    KG_RAG_AVAILABLE = False
    print("⚠️ Knowledge Graph + RAG not available")
    print("   Install: pip install networkx plotly")

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
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    line-height: 1.8;
    color: #1f2937;
    font-size: 1.05rem;
    border: 1px solid #e5e7eb;
}
.answer-box {
    background-color: #eff6ff;
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #3b82f6;
    line-height: 1.8;
    color: #1e40af;
    font-size: 1.05rem;
    border: 1px solid #bfdbfe;
}
.comparison-box {
    background-color: #f0fdf4;
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #22c55e;
    line-height: 1.8;
    color: #065f46;
    font-size: 1.05rem;
    border: 1px solid #bbf7d0;
}
.rag-box {
    background-color: #fef3c7;
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #f59e0b;
    line-height: 1.8;
    color: #92400e;
    font-size: 1.05rem;
    border: 1px solid #fde68a;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.title("🧠 Athena")
st.markdown("### _Local AI Research Assistant powered by Ollama_")

# Show available features
features = []
if COMPARISON_AVAILABLE:
    features.append(" Document Comparison")
if VOICE_AVAILABLE:
    features.append(" Voice Assistant")
if KG_RAG_AVAILABLE:
    features.append(" Knowledge Graph +  Advanced RAG")

if features:
    st.caption(f" Available features: {', '.join(features)}")

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

if COMPARISON_AVAILABLE and "doc_comparison" not in st.session_state:
    st.session_state.doc_comparison = DocumentComparison(model="llama3")

if KG_RAG_AVAILABLE and "advanced_rag" not in st.session_state:
    st.session_state.advanced_rag = AdvancedRAG(chunk_size=800, chunk_overlap=100)

# ---------- Input section ----------
st.markdown("###  Upload a research paper (PDF) or enter a topic")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

topic = st.text_input(
    " Research Topic",
    placeholder="e.g. Recent advances in computer vision"
)

# ---------- Summarization ----------
if st.button(" Research", key="research_button"):
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
                    st.session_state.pdf_filename = uploaded_file.name

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
                    st.session_state.pdf_filename = f"{topic[:30]}.txt"
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
    
    # Build tab list dynamically
    tab_names = [
        "📄 Summary", 
        "💬 Q&A", 
        "🔍 Semantic Search",
        "🤖 Chat with Athena"
    ]
    
    # Add Knowledge Graph + RAG tabs
    if KG_RAG_AVAILABLE:
        tab_names.extend([" Knowledge Graph", " Advanced RAG"])
    
    if COMPARISON_AVAILABLE:
        tab_names.append(" Document Comparison")
    
    if VOICE_AVAILABLE:
        tab_names.append(" Voice Assistant")
    
    tabs = st.tabs(tab_names)
    
    # Tab index counter
    tab_idx = 0

    # ---------- Summary TAB ----------
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown("###  Research Summary")
        st.markdown(f"<div class='result-box'>{st.session_state.last_result}</div>", unsafe_allow_html=True)

        st.download_button(
            label="💾 Download Summary",
            data=st.session_state.last_result,
            file_name="athena_summary.txt",
            mime="text/plain",
            key="download_summary"
        )

    # ---------- Q&A TAB ----------
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown("###  Ask Questions about this Paper")

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
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown("###  Semantic Search in Paper")

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
                with st.spinner(" Performing semantic search..."):
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
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown("###  Chat with Athena")
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
            st.info(" Start a conversation! Ask me anything about research, papers, or AI.")
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
            st.caption(f" {len(st.session_state.chat_messages)} exchanges in this conversation")

    # ---------- KNOWLEDGE GRAPH TAB (NEW) ----------
    if KG_RAG_AVAILABLE:
        with tabs[tab_idx]:
            tab_idx += 1
            render_knowledge_graph_tab(
                st.session_state.pdf_text,
                title=st.session_state.get('pdf_filename', 'Research Document')
            )

    # ---------- ADVANCED RAG TAB (NEW) ----------
    if KG_RAG_AVAILABLE:
        with tabs[tab_idx]:
            tab_idx += 1
            st.markdown("###  Advanced Multi-Document RAG")
            st.info("💡 Ask questions across multiple documents with source attribution and confidence scoring")
            
            rag = st.session_state.advanced_rag
            
            # Add current document to RAG
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("#### 📎 Manage Documents")
            
            with col2:
                if st.button("🗑️ Clear All", key="clear_rag"):
                    st.session_state.advanced_rag = AdvancedRAG(chunk_size=800)
                    st.success("RAG system cleared!")
                    st.rerun()
            
            if st.button("➕ Add Current Document to RAG", type="primary", key="add_to_rag"):
                doc_id = st.session_state.get('pdf_filename', 'document_1')
                
                with st.spinner("Adding document to RAG system..."):
                    rag.add_document(
                        doc_id=doc_id,
                        title=doc_id,
                        content=st.session_state.pdf_text,
                        metadata={'type': 'research_paper'}
                    )
                st.success(f"✅ Added: {doc_id}")
                st.rerun()
            
            # Show loaded documents
            summary = rag.get_document_summary()
            
            if summary['total_documents'] > 0:
                st.markdown(f"** Loaded Documents:** {summary['total_documents']}")
                
                for doc in summary['documents']:
                    with st.expander(f" {doc['title']}", expanded=False):
                        st.write(f"**Length:** {doc['length']:,} characters")
                        st.write(f"**ID:** {doc['id']}")
                        if doc.get('metadata'):
                            st.write(f"**Metadata:** {doc['metadata']}")
                
                st.markdown("---")
                
                # Query interface
                st.markdown("####  Ask Questions Across Documents")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    rag_query = st.text_input(
                        "Ask a question",
                        placeholder="e.g., How do these papers approach attention mechanisms?",
                        key="rag_query_input"
                    )
                
                with col2:
                    k_contexts = st.number_input("Contexts", min_value=3, max_value=10, value=5, key="k_contexts")
                
                # Document filter (optional)
                if summary['total_documents'] > 1:
                    doc_filter = st.multiselect(
                        "Filter by documents (leave empty for all)",
                        options=[doc['id'] for doc in summary['documents']],
                        key="doc_filter"
                    )
                else:
                    doc_filter = None
                
                if st.button(" Answer with RAG", type="primary", key="rag_answer"):
                    if rag_query:
                        with st.spinner(" Retrieving context and generating answer..."):
                            result = rag.answer_with_context(
                                rag_query,
                                k=k_contexts,
                                doc_ids=doc_filter if doc_filter else None
                            )
                        
                        # Display answer
                        st.markdown("### 💡 Answer")
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**Confidence:** {result['confidence']:.0%}")
                        with col2:
                            st.markdown(f"**Sources Used:** {result['num_sources_used']}")
                        
                        st.markdown(f"<div class='rag-box'>{result['answer']}</div>", unsafe_allow_html=True)
                        
                        # Display sources
                        st.markdown("###  Sources")
                        for source in result['sources']:
                            with st.expander(
                                f"[{source['source_id']}] {source['doc_title']} "
                                f"(Similarity: {source['similarity']:.0%})",
                                expanded=(source['source_id'] <= 2)
                            ):
                                st.write(f"**Document ID:** {source['doc_id']}")
                                st.write(f"**Chunk:** {source['chunk_index']}")
                                st.write(f"**Relevance:** {source['similarity']:.1%}")
                    else:
                        st.warning("Please enter a question")
                
                # Comparison feature
                if summary['total_documents'] >= 2:
                    st.markdown("---")
                    st.markdown("###  Compare Documents")
                    st.info(" Compare multiple documents on a specific topic to find similarities and differences")
                    
                    comparison_query = st.text_input(
                        "Topic to compare",
                        placeholder="e.g., training methodology, attention mechanisms, evaluation metrics",
                        key="comparison_query_input"
                    )
                    
                    # Select documents to compare
                    docs_to_compare = st.multiselect(
                        "Select documents to compare (2+)",
                        options=[doc['id'] for doc in summary['documents']],
                        default=[doc['id'] for doc in summary['documents'][:min(3, len(summary['documents']))]],
                        key="docs_to_compare"
                    )
                    
                    if st.button(" Compare Documents", key="compare_docs"):
                        if comparison_query and len(docs_to_compare) >= 2:
                            with st.spinner(" Analyzing differences and similarities..."):
                                comp_result = rag.compare_documents(
                                    comparison_query,
                                    doc_ids=docs_to_compare,
                                    k=3
                                )
                            
                            if 'comparison' in comp_result:
                                st.markdown("###  Comparison Analysis")
                                st.markdown(f"<div class='comparison-box'>{comp_result['comparison']}</div>", unsafe_allow_html=True)
                                
                                st.markdown("**Documents Compared:**")
                                for doc in comp_result['documents_compared']:
                                    st.write(f"- {doc}")
                            else:
                                st.error(f"Error: {comp_result.get('error', 'Unknown error')}")
                        else:
                            if not comparison_query:
                                st.warning("Please enter a topic to compare")
                            elif len(docs_to_compare) < 2:
                                st.warning("Please select at least 2 documents to compare")
                
                # Concept tracking
                if summary['total_documents'] >= 2:
                    st.markdown("---")
                    st.markdown("###  Track Concepts Across Documents")
                    st.info(" See how a specific concept appears and evolves across multiple papers")
                    
                    concept_query = st.text_input(
                        "Concept to track",
                        placeholder="e.g., transfer learning, attention, CNN",
                        key="concept_query_input"
                    )
                    
                    if st.button(" Track Concept", key="track_concept"):
                        if concept_query:
                            with st.spinner(" Tracing concept across documents..."):
                                conn_result = rag.find_connections(concept_query)
                            
                            if 'analysis' in conn_result:
                                st.markdown("###  Concept Analysis")
                                
                                st.markdown(f"**Concept:** {conn_result['concept']}")
                                st.markdown(f"**Found in:** {len(conn_result['documents'])} documents")
                                st.markdown(f"**Total mentions:** {conn_result['total_mentions']}")
                                
                                st.markdown("---")
                                st.markdown(conn_result['analysis'])
                                
                                st.markdown("**Documents:**")
                                for doc in conn_result['documents']:
                                    st.write(f"- {doc}")
                            else:
                                st.error(f"Error: {conn_result.get('error', 'Unknown error')}")
                        else:
                            st.warning("Please enter a concept to track")
            
            else:
                st.warning("⚠️ No documents loaded. Add the current document using the button above.")
                st.markdown("**📖 How to use Advanced RAG:**")
                st.markdown("1. Upload and research a paper")
                st.markdown("2. Click '➕ Add Current Document to RAG'")
                st.markdown("3. Upload more papers and add them")
                st.markdown("4. Ask questions across all papers")
                st.markdown("5. Compare documents or track concepts")

    # ---------- DOCUMENT COMPARISON TAB (Optional) ----------
    if COMPARISON_AVAILABLE:
        with tabs[tab_idx]:
            tab_idx += 1
            st.markdown("###  Document Comparison")
            st.info("💡 Upload multiple PDFs to compare them side-by-side and find similarities/differences!")
            
            st.markdown("#### 📎 Upload Documents to Compare")
            
            col1, col2 = st.columns(2)
            
            with col1:
                doc1_file = st.file_uploader("Document 1", type="pdf", key="doc1_upload")
            
            with col2:
                doc2_file = st.file_uploader("Document 2", type="pdf", key="doc2_upload")
            
            if st.button(" Compare Documents", key="compare_button"):
                if not doc1_file or not doc2_file:
                    st.warning("⚠️ Please upload both documents to compare.")
                else:
                    with st.spinner("📖 Extracting and analyzing documents..."):
                        try:
                            text1 = extract_text_from_pdf(doc1_file)
                            text2 = extract_text_from_pdf(doc2_file)
                            
                            if not text1.strip() or not text2.strip():
                                st.error("❌ Could not extract text from one or both PDFs.")
                                st.stop()
                            
                            st.session_state.doc_comparison.add_document(doc1_file.name, text1)
                            st.session_state.doc_comparison.add_document(doc2_file.name, text2)
                            
                            st.success("✅ Documents loaded successfully!")
                            
                            with st.spinner(" Comparing documents..."):
                                comparison_result = st.session_state.doc_comparison.compare_documents(
                                    doc1_file.name,
                                    doc2_file.name
                                )
                                st.session_state.comparison_result = comparison_result
                            
                        except Exception as e:
                            st.error(f"❌ Error during comparison: {e}")
            
            if "comparison_result" in st.session_state and st.session_state.comparison_result:
                st.markdown("---")
                st.markdown("###  Comparison Results")
                
                result = st.session_state.comparison_result
                
                st.markdown("####  Summary")
                st.markdown(f"<div class='comparison-box'>{result['summary']}</div>", unsafe_allow_html=True)
                
                st.markdown("####  Similarities")
                with st.expander("View Common Topics", expanded=True):
                    st.markdown(f"<div class='result-box'>{result['similarities']}</div>", unsafe_allow_html=True)
                
                st.markdown("####  Key Differences")
                with st.expander("View Unique Aspects", expanded=True):
                    st.markdown(f"<div class='result-box'>{result['differences']}</div>", unsafe_allow_html=True)
                
                if result.get('recommendations'):
                    st.markdown("####  Recommendations")
                    with st.expander("View Insights"):
                        st.markdown(f"<div class='result-box'>{result['recommendations']}</div>", unsafe_allow_html=True)
                
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

    # ---------- VOICE ASSISTANT TAB (Optional) ----------
    if VOICE_AVAILABLE:
        with tabs[tab_idx]:
            tab_idx += 1
            render_voice_tab()

# ---------- Footer ----------
st.markdown("---")
st.caption("Athena © 2025 · Built with Streamlit & LangChain")

# ---------- Sidebar Info ----------
with st.sidebar:
    st.markdown("### 🧠 Athena Features")
    
    st.markdown("#### Core Features")
    st.markdown("-  Document Summarization")
    st.markdown("-  Q&A System")
    st.markdown("-  Semantic Search")
    st.markdown("-  Chat Interface")
    
    if KG_RAG_AVAILABLE:
        st.markdown("#### Advanced Features")
        st.markdown("-  Knowledge Graphs")
        st.markdown("-  Multi-Document RAG")
        st.markdown("-  Cross-Paper Comparison")
        st.markdown("-  Concept Tracking")
    
    if COMPARISON_AVAILABLE or VOICE_AVAILABLE:
        st.markdown("#### Optional Features")
        if COMPARISON_AVAILABLE:
            st.markdown("-  Document Comparison")
        if VOICE_AVAILABLE:
            st.markdown("-  Voice Assistant")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Athena** is a local AI research assistant that helps you:
    - Analyze research papers
    - Extract key insights
    - Build knowledge graphs
    - Compare multiple documents
    - Ask contextual questions
    
    Powered by Ollama (llama3) and running entirely on your machine.
    """)
    
    st.markdown("---")
    st.markdown("### 🔧 System Status")
    
    # Check Ollama status
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            st.success("✅ Ollama: Running")
            models = response.json().get('models', [])
            if any('llama3' in m.get('name', '') for m in models):
                st.success("✅ llama3: Available")
            else:
                st.warning("⚠️ llama3: Not found")
        else:
            st.error("❌ Ollama: Error")
    except:
        st.error("❌ Ollama: Not running")
        st.caption("Start: `ollama serve`")
    
    # Session stats
    if st.session_state.pdf_uploaded:
        st.markdown("---")
        st.markdown("###  Session Stats")
        st.write(f"**Document:** {st.session_state.get('pdf_filename', 'N/A')}")
        st.write(f"**Text length:** {len(st.session_state.pdf_text):,} chars")
        
        if 'qa_chain' in st.session_state:
            st.write("**Q&A Index:** ✅ Ready")
        
        if 'semantic_index' in st.session_state:
            st.write("**Semantic Index:** ✅ Ready")
        
        if KG_RAG_AVAILABLE and 'kg_builder' in st.session_state:
            summary = st.session_state.kg_builder.get_graph_summary()
            st.write(f"**Knowledge Graph:** {summary['total_nodes']} nodes")
        
        if KG_RAG_AVAILABLE:
            rag_summary = st.session_state.advanced_rag.get_document_summary()
            st.write(f"**RAG Documents:** {rag_summary['total_documents']}")
    
    st.markdown("---")
    st.markdown("### 💡 Quick Tips")
    
    tips = [
        " Start by uploading a PDF or entering a topic",
        " Use Q&A for specific factual questions",
        " Use Semantic Search to explore topics",
        " Use Chat for conversational exploration",
    ]
    
    if KG_RAG_AVAILABLE:
        tips.extend([
            " Build Knowledge Graphs to visualize entities",
            " Add multiple papers to RAG for comparison",
            " Track concepts across documents"
        ])
    
    for tip in tips:
        st.caption(tip)
    
    st.markdown("---")
    
    # Reset button
    if st.button("🔄 Reset Session", key="reset_session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("Session reset! Refresh the page.")
        st.rerun()