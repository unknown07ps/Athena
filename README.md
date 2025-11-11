# 🧠 Athena - AI Research Assistant

A powerful local AI research assistant powered by Ollama and LangChain. Athena helps you analyze research papers, perform semantic search, get intelligent answers, and have natural conversations about your documents.

![Athena Banner](https://img.shields.io/badge/AI-Research_Assistant-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

## ✨ Key Features

Athena provides **four powerful research modes**, each designed for different research needs:

### 📄 Summary Mode
- **What it does:** Generates comprehensive summaries of research papers or documents
- **Best for:** Getting quick overviews, understanding main concepts
- **How it works:** Uses AI to extract and synthesize key information into a cohesive summary
- **When to use:** 
  - First-time reading of a paper
  - Quick literature review
  - Understanding document structure

### 💬 Q&A Mode
- **What it does:** Answers specific, targeted questions about your document
- **Best for:** Finding precise information, clarifying specific points
- **How it works:** 
  - Builds a FAISS vector index of your document
  - Uses RAG (Retrieval-Augmented Generation) to find relevant sections
  - Generates answers based on retrieved context
- **When to use:**
  - Looking for specific facts or data
  - Understanding particular sections
  - Extracting technical details
- **Example questions:**
  - "What dataset was used in this study?"
  - "What were the main results?"
  - "How did the authors evaluate their model?"

### 🔍 Semantic Search
- **What it does:** Finds all relevant sections using natural language queries
- **Best for:** Exploring topics, finding all mentions, discovering related content
- **How it works:**
  - Creates semantic embeddings of document chunks
  - Ranks results by similarity to your query
  - Shows multiple matching sections with relevance scores
- **When to use:**
  - Exploring a concept throughout the paper
  - Finding all related mentions
  - Comparing different sections
- **Example searches:**
  - "experimental methodology"
  - "limitations and future work"
  - "related work and citations"

### 🤖 Chat with Athena
- **What it does:** Natural, conversational interaction about your document
- **Best for:** In-depth discussion, follow-up questions, contextual understanding
- **How it works:**
  - Maintains conversation history
  - Has full document context
  - Provides nuanced, context-aware responses
- **When to use:**
  - Need multiple related questions answered
  - Want to discuss and explore ideas
  - Prefer conversational interaction
- **Example conversations:**
  - "Explain the transformer architecture mentioned here"
  - "How does this compare to previous approaches?"
  - "What are the implications of these findings?"

### 📊 Document Comparison
- **What it does:** Compares multiple PDFs side-by-side to find similarities and differences
- **Best for:** Literature reviews, choosing between papers, understanding relationships
- **How it works:**
  - Extracts text from both documents
  - Calculates semantic similarity using embeddings
  - Uses LLM to generate detailed comparative analysis
  - Identifies common themes and unique aspects
- **When to use:**
  - Comparing research papers on similar topics
  - Deciding which paper to cite
  - Understanding how approaches differ
  - Literature review synthesis
- **Example use cases:**
  - "Compare two transformer papers"
  - "Find differences between CNN architectures"
  - "Which paper is more relevant to my research?"

## 🔄 When to Use Each Mode

| Mode | Use Case | Speed | Depth | Best For |
|------|----------|-------|-------|----------|
| **Summary** | First overview | ⚡ Fast | 📊 Broad | Getting started |
| **Q&A** | Specific questions | 🚀 Medium | 🎯 Precise | Finding facts |
| **Semantic Search** | Find all mentions | 🔍 Fast | 📚 Comprehensive | Exploring topics |
| **Chat** | Discussion | 💬 Interactive | 🧠 Deep | Understanding concepts |
| **Comparison** | Compare papers | 📊 Medium | 🔬 Analytical | Literature review |

### Comparison Example

Let's say you're reading a machine learning paper:

**Summary:** "This paper introduces a new attention mechanism..."
- ✅ Great for: Initial understanding
- ❌ Not ideal for: Specific technical details

**Q&A:** "What was the training batch size?"
- ✅ Great for: Extracting specific values
- ❌ Not ideal for: Exploring related concepts

**Semantic Search:** Search for "training procedure"
- ✅ Great for: Finding all training-related sections
- ❌ Not ideal for: Getting one specific answer

**Chat:** "Walk me through how they trained the model"
- ✅ Great for: Understanding the full process
- ❌ Not ideal for: Quick facts

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Ollama installed and running

### 1. Install Ollama

Download and install Ollama from [ollama.ai](https://ollama.ai)

```bash
# Pull the Llama 3 model (required)
ollama pull llama3

# Start Ollama server
ollama serve
```

### 2. Clone the Repository

```bash
git clone <your-repo-url>
cd athena
```

### 3. Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🎯 Usage

### Quick Start

```bash
# 1. Make sure Ollama is running
ollama serve

# 2. Start Athena
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Step-by-Step Guide

#### 1. Upload a Document
- Click "📎 Upload PDF" and select your research paper
- Or enter a topic in "🔍 Research Topic" for web-based research
- Click "✨ Start Research"

#### 2. Choose Your Research Mode

**For Quick Overview:**
- Switch to "📄 Summary" tab
- Read the comprehensive summary
- Download with "💾 Download Summary"

**For Specific Questions:**
- Switch to "💬 Q&A" tab
- Wait for index to build (one-time setup)
- Type your question and click "🚀 Get Answer"
- Example: "What methodology was used?"

**For Exploring Topics:**
- Switch to "🔍 Semantic Search" tab
- Enter keywords or phrases
- Adjust similarity threshold with slider
- View ranked results with relevance scores

**For Conversations:**
- Switch to "🤖 Chat with Athena" tab
- Start a natural conversation
- Ask follow-up questions
- Export conversation with "💾 Export Chat"

## 🛠️ Project Structure

```
athena/
├── app.py                    # Main Streamlit application (Enhanced UI)
├── main.py                   # Research logic (online/offline)
├── qa_engine.py              # Q&A system with FAISS & RAG
├── semantic_search.py        # Semantic search with embeddings
├── chat_engine.py            # Conversational AI with context
├── pdf_utils.py              # PDF text extraction & cleaning
├── tools/
│   ├── arxiv_search.py      # Arxiv paper search
│   └── web_search.py        # DuckDuckGo web search
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🔧 Configuration

### Change the LLM Model

Edit the model parameter in `app.py`, `qa_engine.py`, `chat_engine.py`, and `main.py`:

```python
# Change from llama3 to another model
model="mistral"  # or "llama2", "codellama", "phi", etc.
```

### Adjust Performance Settings

**Q&A System** (`qa_engine.py`):
```python
chunk_size=2000      # Larger = more context, slower
chunk_overlap=200    # Overlap between chunks
k=3                  # Number of chunks to retrieve
```

**Semantic Search** (`semantic_search.py`):
```python
chunk_size=300       # Smaller = more precise results
chunk_overlap=50     # Less overlap for speed
k=10                 # Number of results to show
```

**Chat Engine** (`chat_engine.py`):
```python
temperature=0.3      # Lower = more focused, higher = more creative
num_predict=500      # Max response length
```

## 🐛 Troubleshooting

### "Could not connect to Ollama"
```bash
# Check if Ollama is running
ollama list

# Start Ollama server
ollama serve

# Verify the URL is correct
curl http://localhost:11434/api/tags
```

### "No text extracted from PDF"
- PDF might be scanned (image-based)
- Try OCR tools: `pip install pdfplumber`
- Check if PDF is encrypted or password-protected

### "Import errors"
```bash
# Reinstall all dependencies
pip install --upgrade -r requirements.txt

# Or install individually
pip install streamlit langchain faiss-cpu sentence-transformers
```

### Slow Performance

**Option 1: Use a smaller model**
```bash
ollama pull llama2:7b
# Update model="llama2:7b" in code
```

**Option 2: Reduce chunk sizes**
```python
# In qa_engine.py and semantic_search.py
chunk_size=1000  # Smaller value = faster
k=2              # Fewer results = faster
```

**Option 3: Optimize embeddings**
```python
# Use a faster embedding model
embed_model = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"  # Faster
)
```

### Q&A Not Working
```bash
# Make sure FAISS is installed correctly
pip uninstall faiss-cpu
pip install faiss-cpu

# Test FAISS
python -c "import faiss; print('FAISS OK')"
```

### Chat Giving Wrong Answers
- Make sure PDF was uploaded successfully
- Check if Ollama model is loaded: `ollama list`
- Try reducing temperature: `temperature=0.1`
- Increase context in `chat_engine.py`

## 📦 Dependencies

### Core
- **streamlit** - Web framework
- **langchain** - LLM orchestration
- **ollama** - Local LLM runtime

### AI/ML
- **faiss-cpu** - Vector similarity search
- **sentence-transformers** - Text embeddings
- **PyPDF2** - PDF text extraction

### Search Tools
- **duckduckgo-search** - Web search
- **arxiv** - Academic paper search

## 🎨 UI Features

- **Modern gradient design** with purple/blue theme
- **Responsive layout** that works on all screen sizes
- **Color-coded results** (🟢 High, 🟡 Medium, 🟠 Low relevance)
- **Smooth animations** and transitions
- **Dark mode compatible** styling
- **Accessible** with proper contrast ratios

## 🚀 Advanced Features

### Web Research Mode
Enter a topic without uploading a PDF to search the web and Arxiv:
```
Topic: "Recent advances in transformer models"
```
Athena will:
1. Search DuckDuckGo for recent articles
2. Search Arxiv for academic papers
3. Synthesize findings into a comprehensive summary

### Batch Processing
Process multiple PDFs by running Athena in a loop (advanced users):
```python
# custom_batch.py
from pdf_utils import extract_text_from_pdf
from semantic_search import build_semantic_index

pdfs = ["paper1.pdf", "paper2.pdf", "paper3.pdf"]
for pdf in pdfs:
    text = extract_text_from_pdf(pdf)
    index = build_semantic_index(text)
    # Process...
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional LLM providers (OpenAI, Anthropic)
- OCR support for scanned PDFs
- Multi-language support
- Citation extraction
- Graph visualization of concepts

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai) - Local LLM runtime
- [LangChain](https://langchain.com) - LLM framework
- [Streamlit](https://streamlit.io) - Web framework
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [Sentence Transformers](https://www.sbert.net/) - Embeddings

## 📧 Support

Having issues? Here's how to get help:

1. **Check troubleshooting section** above
2. **Run the test script**: `python test_system.py`
3. **Check Ollama status**: `ollama list`
4. **Open an issue** on GitHub with error details

## 📊 Performance Tips

### For Large Documents (>100 pages)
- Increase chunk size to 3000-4000
- Reduce number of chunks retrieved (k=2)
- Use summary mode first

### For Quick Responses
- Use smaller LLM models (llama2:7b)
- Reduce temperature to 0.1
- Limit max tokens to 250

### For Best Quality
- Use llama3 or larger models
- Increase k to 5 for Q&A
- Use temperature 0.3-0.5

---

## 🎯 Quick Reference

| Task | Mode | Command |
|------|------|---------|
| Understand paper | Summary | Click "Start Research" |
| Find specific info | Q&A | Ask targeted question |
| Explore topic | Semantic Search | Search with keywords |
| Deep discussion | Chat | Start conversation |
| Save results | Any mode | Use download buttons |

**Built with ❤️ for researchers and students**

*Making research accessible, one paper at a time* 🚀