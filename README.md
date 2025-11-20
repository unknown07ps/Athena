# Athena - AI Research Assistant

A powerful local AI research assistant powered by Ollama and LangChain. Automatically fetch research papers from arXiv and Semantic Scholar, analyze documents, build knowledge graphs, perform multi-document reasoning, and more—all running locally on your machine.

---

## Features

### Core Research Tools
- **Automated Paper Fetching** - Search and retrieve papers from arXiv and Semantic Scholar automatically
- **Document Summarization** - Intelligent multi-section summarization with source citations
- **Q&A System** - RAG-based question answering with context
- **Semantic Search** - Find relevant sections using natural language
- **Chat Interface** - Conversational AI with document context
- **Document Comparison** - Deep comparative analysis of multiple papers

### Advanced Features
- **Knowledge Graph Construction** - Automatically extract and visualize entities, relationships, and concepts
- **Multi-Document RAG** - Cross-paper reasoning with source attribution and confidence scoring
- **Concept Tracking** - Trace how concepts evolve across different papers
- **Voice Interface** - Speech-to-text and text-to-speech capabilities (optional)
- **Performance Metrics** - Track entity relationships and research trends

### Use Cases
- **Literature Reviews** - Automatically gather and synthesize relevant papers
- **Research Analysis** - Extract entities, methods, datasets, and results
- **Academic Writing** - Find relevant citations and conceptual connections
- **Paper Understanding** - Visual knowledge graphs and contextual Q&A
- **Research Discovery** - Stay updated with latest papers in your field

---

## Quick Start

### Prerequisites

1. **Python 3.8 or higher**
   ```bash
   python --version  # Should show 3.8+
   ```

2. **Ollama** (for local LLM)
   - Download from [ollama.ai](https://ollama.ai)
   - Install and run:
   ```bash
   ollama pull llama3
   ollama serve
   ```

### Installation

#### Option 1: Automated Setup (Recommended)

**Linux/macOS:**
```bash
git clone https://github.com/yourusername/athena.git
cd athena
chmod +x setup.sh
./setup.sh
```

**Windows:**
```bash
git clone https://github.com/yourusername/athena.git
cd athena
setup.bat
```

#### Option 2: Manual Setup

```bash
# 1. Clone repository
git clone https://github.com/yourusername/athena.git
cd athena

# 2. Create virtual environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install optional features (recommended)
pip install -r requirements_optional.txt

# 5. Verify installation
python check_setup.py
```

### Running Athena

```bash
# Make sure Ollama is running
ollama serve

# Start Athena
streamlit run app.py
```

Access at: `http://localhost:8501`

---

## Core Dependencies

```txt
# Core requirements
streamlit>=1.28.0
PyPDF2>=3.0.0
langchain>=0.1.0
langchain-community>=0.0.10
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
requests>=2.31.0
numpy>=1.24.0
scikit-learn>=1.3.0

# For paper fetching
arxiv>=2.0.0
semanticscholar>=0.3.0

# Advanced features (optional)
networkx>=3.0              # Knowledge graphs
plotly>=5.18.0            # Visualizations
pyvis>=0.3.2              # Interactive graphs
faster-whisper>=0.9.0     # Voice (STT)
gtts>=2.4.0               # Voice (TTS)
```

---

## Usage Examples

### 1. Fetch and Analyze Research Papers

Athena automatically fetches papers from online sources.

```bash
# Start Athena
streamlit run app.py

# In the web interface:
# 1. Enter research topic: "transformer attention mechanisms in NLP"
# 2. Click "Research"
# 3. Athena will:
#    - Search arXiv and Semantic Scholar
#    - Retrieve top 5 most relevant papers
#    - Analyze abstracts and metadata
#    - Generate comprehensive summary with citations
#    - Provide links to full papers and PDFs
```

**Example Topics to Try:**
- "Recent advances in computer vision"
- "Large language models for code generation"
- "Graph neural networks applications"
- "Federated learning privacy"
- "Multimodal transformers"

**What You Get:**
- Synthesized overview of the topic
- Key findings from each paper
- Common themes and methodologies
- Recent advances and breakthroughs
- Challenges and future directions
- Direct links to papers and PDFs
- Paper metadata (authors, year, citations)

### 2. Analyze a Single Research Paper

```bash
# In the web interface:
# 1. Upload your PDF
# 2. Click "Research"
# 3. Explore different tabs:
#    - Summary: High-level overview
#    - Q&A: Ask specific questions
#    - Search: Find relevant sections
#    - Chat: Conversational exploration
```

### 3. Build Knowledge Graph

```python
# In Athena web interface:
# 1. Upload/research a paper
# 2. Go to "Knowledge Graph" tab
# 3. Click "Build Knowledge Graph"
# 4. Explore:
#    - Interactive visualization (drag, zoom, hover)
#    - Entity queries (search for concepts)
#    - Path finding (connections between entities)
#    - Export options (JSON, GraphML)
```

**Extracted Entities:**
- Papers and authors
- Methods and algorithms
- Datasets and benchmarks
- Metrics and results
- Model architectures

### 4. Multi-Document RAG & Comparison

```python
# In Athena web interface:
# 1. Research topic → get 5 papers automatically
# 2. Go to "Advanced RAG" tab
# 3. Click "Add Current Document to RAG"
# 4. Repeat for multiple papers
# 5. Use advanced features:
#    - Ask cross-paper questions
#    - Compare methodologies
#    - Track concepts across papers
#    - Get answers with source attribution
```

**Example Queries:**
- "How do these papers approach attention mechanisms?"
- "What datasets are commonly used?"
- "Compare the performance metrics across papers"
- "Which paper has the best results on ImageNet?"

### 5. Voice Interaction (Optional)

```python
# Prerequisites:
# pip install faster-whisper gtts

# In Athena web interface:
# 1. Go to "Voice Assistant" tab
# 2. Record your question
# 3. Get spoken response
# 4. View transcription and answer
```

---

## Project Structure

```
athena/
├── app.py                          # Main Streamlit application
├── main.py                         # Research engine with paper fetching
├── paper_fetcher.py               # Paper search from arXiv & Semantic Scholar
├── qa_engine.py                    # Q&A system with FAISS
├── semantic_search.py              # Semantic search engine
├── chat_engine.py                  # Conversational AI
├── pdf_utils.py                    # PDF extraction utilities
│
├── advanced_rag.py                 # Multi-document RAG system
├── knowledge_graph.py              # Knowledge graph construction
├── kg_visualizer.py               # Graph visualization
├── document_comparison.py          # Document comparison engine
│
├── voice_engine.py                 # Voice processing (optional)
├── voice_interface.py             # Voice UI integration (optional)
│
├── tools/
│   ├── arxiv_search.py            # Arxiv paper search
│   └── web_search.py              # DuckDuckGo web search
│
├── docs/                           # Documentation
│   ├── INSTALLATION.md
│   ├── QUICKSTART.md
│   ├── FEATURES.md
│   ├── KNOWLEDGE_GRAPH_GUIDE.md
│   ├── ADVANCED_RAG_GUIDE.md
│   ├── VOICE_INTERFACE_GUIDE.md
│   ├── TROUBLESHOOTING.md
│   └── API.md
│
├── tests/                          # Test scripts
│   ├── test_system.py             # Core system tests
│   ├── test_paper_fetcher.py     # Paper fetching tests
│   ├── test_kg_rag_system.py     # KG + RAG tests
│   ├── test_comparison.py         # Document comparison tests
│   └── test_voice.py              # Voice interface tests
│
├── requirements.txt                # Core dependencies
├── requirements_optional.txt       # Optional features
├── setup.sh                        # Linux/macOS setup
├── setup.bat                       # Windows setup
├── check_setup.py                  # Installation verifier
│
└── README.md                       # This file
```

---

## Configuration

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB+ |
| Storage | 5 GB | 10 GB+ |
| CPU | 4 cores | 8+ cores |
| GPU | Not required | NVIDIA GPU (optional) |
| Internet | For paper fetching | Stable connection |

### Ollama Models

**Default:** `llama3` (7B parameters)

**Alternatives:**
```bash
# Smaller (faster, less accurate)
ollama pull llama2:7b

# Larger (slower, more accurate)
ollama pull llama3:70b

# Specialized
ollama pull mistral
ollama pull codellama
```

### Paper Fetching Configuration

**In `main.py`:**
```python
# Adjust maximum papers to fetch
result = research_topic(
    topic="your topic",
    fetch_papers=True,
    max_papers=5  # Change to 3, 10, etc.
)
```

**In `paper_fetcher.py`:**
```python
# Configure search sources
papers = fetcher.search_papers(
    query=topic,
    max_results=5,
    sources=['arxiv', 'semantic_scholar']  # Add/remove sources
)
```

### Performance Tuning

**For faster processing:**
```python
# In qa_engine.py, semantic_search.py
chunk_size = 1000  # Smaller = faster
k = 2              # Fewer results = faster
max_papers = 3     # Fewer papers = faster
```

**For better quality:**
```python
# In qa_engine.py, semantic_search.py
chunk_size = 3000  # Larger = more context
k = 5              # More results = better coverage
temperature = 0.1  # Lower = more focused
max_papers = 10    # More papers = comprehensive
```

---

## Testing

### Run All Tests
```bash
# Core system test
python tests/test_system.py

# Paper fetching test
python tests/test_paper_fetcher.py

# Knowledge Graph + RAG test
python tests/test_kg_rag_system.py

# Document comparison test
python tests/test_comparison.py

# Voice interface test (if installed)
python tests/test_voice.py
```

### Quick Verification
```bash
# Verify installation
python check_setup.py

# Check Ollama status
curl http://localhost:11434/api/tags

# Test paper fetching
python -c "from paper_fetcher import PaperFetcher; f=PaperFetcher(); print(len(f.search_papers('transformer', 3)))"
```

---

## How Paper Fetching Works

### Architecture

```
User Input (Topic)
        ↓
┌───────────────────────┐
│   main.py             │
│   research_topic()    │
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│  paper_fetcher.py     │
│  PaperFetcher class   │
└───────────┬───────────┘
            ↓
    ┌───────┴───────┐
    ↓               ↓
┌─────────┐   ┌──────────────┐
│  arXiv  │   │ Semantic     │
│   API   │   │ Scholar API  │
└────┬────┘   └──────┬───────┘
     │               │
     └───────┬───────┘
             ↓
    ┌────────────────┐
    │ Paper Metadata │
    │ + Abstracts    │
    └────────┬───────┘
             ↓
    ┌────────────────┐
    │  Ollama LLM    │
    │  (Analysis &   │
    │  Synthesis)    │
    └────────┬───────┘
             ↓
    ┌────────────────┐
    │ Comprehensive  │
    │ Summary with   │
    │ Citations      │
    └────────────────┘
```

### Search Strategy

1. **Query Optimization**: Convert user topic to search-friendly query
2. **Multi-Source Search**: 
   - arXiv: Academic preprints (CS, Physics, Math, etc.)
   - Semantic Scholar: Published papers with citation data
3. **Deduplication**: Remove duplicate papers across sources
4. **Ranking**: Sort by relevance and recency
5. **Metadata Extraction**: Authors, year, citations, abstracts
6. **Synthesis**: LLM analyzes and summarizes findings

### Supported Sources

| Source | Coverage | Features |
|--------|----------|----------|
| **arXiv** | 2M+ preprints | Latest research, free PDFs |
| **Semantic Scholar** | 200M+ papers | Citation counts, metadata |

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# 1. Fork and clone
git clone https://github.com/yourusername/athena.git
cd athena

# 2. Create branch
git checkout -b feature/your-feature

# 3. Install dev dependencies
pip install -r requirements_dev.txt

# 4. Make changes and test
python -m pytest tests/

# 5. Submit pull request
```

### Areas for Contribution
- Additional paper sources (PubMed, IEEE, ACM)
- Additional LLM providers (OpenAI, Anthropic, Claude)
- OCR support for scanned PDFs
- Multi-language support
- Enhanced visualizations
- Additional test coverage
- Improved documentation

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Built with these amazing open-source projects:

- [Ollama](https://ollama.ai) - Local LLM runtime
- [LangChain](https://langchain.com) - LLM framework
- [Streamlit](https://streamlit.io) - Web framework
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [Sentence Transformers](https://www.sbert.net/) - Embeddings
- [NetworkX](https://networkx.org/) - Graph analysis
- [Plotly](https://plotly.com/) - Interactive visualizations
- [arXiv API](https://arxiv.org/help/api) - Academic paper search
- [Semantic Scholar API](https://www.semanticscholar.org/product/api) - Citation data

---

## Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/athena/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/athena/discussions)
- **Email:** your.email@example.com

---

## Roadmap

### v2.0 (Current) - Completed
- Automated paper fetching from arXiv
- Semantic Scholar integration
- Multi-document RAG
- Knowledge graph visualization
- Document comparison

### v2.1 (In Progress)
- Additional sources (PubMed, IEEE, ACM Digital Library)
- Citation network visualization
- Research trend analysis
- Collaborative features (shared workspaces)
- Export to reference managers (Zotero, Mendeley)

### v2.2 (Planned)
- Neo4j integration for large knowledge graphs
- LangGraph multi-agent workflows
- Cloud deployment options
- API server mode
- Custom LLM fine-tuning support

### v3.0 (Future)
- Graph neural networks for paper similarity
- Temporal concept tracking
- Automated literature review generation
- Research gap identification
- Hypothesis generation

---

## Recent Updates

### v2.0 - Paper Fetching Release (Latest)
- NEW: Automated paper fetching from arXiv and Semantic Scholar
- NEW: Comprehensive research synthesis with source citations
- NEW: Direct links to papers and PDFs
- Improved multi-document RAG performance
- Enhanced knowledge graph entity extraction
- Bug fixes and stability improvements

---

**Built with care for researchers and students**

*Making research accessible, one paper at a time*

---

## Quick Tips

1. **Start with a broad topic** to get an overview, then dive deeper
2. **Use Knowledge Graphs** to visualize connections between concepts
3. **Add multiple papers to RAG** for cross-paper analysis
4. **Adjust `max_papers`** based on your needs (3-10 recommended)
5. **Save interesting papers** by downloading the summary with citations
6. **Combine features**: Fetch papers → Build KG → Ask questions → Compare
7. **Check paper citations** to assess impact and relevance

---

**Questions? Open an issue or start a discussion!**
