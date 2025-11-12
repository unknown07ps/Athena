#!/bin/bash
# setup_kg_rag.sh - Quick setup for Knowledge Graph + RAG features

echo "============================================================"
echo "🧠 ATHENA KNOWLEDGE GRAPH + RAG SETUP"
echo "============================================================"
echo ""

# Check Python version
echo "1️⃣ Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "   Found: Python $python_version"

if [[ $(echo "$python_version" | cut -d. -f1) -lt 3 ]] || [[ $(echo "$python_version" | cut -d. -f2) -lt 8 ]]; then
    echo "   ❌ Python 3.8+ required"
    exit 1
fi
echo "   ✅ Python version OK"
echo ""

# Install core dependencies
echo "2️⃣ Installing core dependencies..."
echo "   This may take 2-3 minutes..."
echo ""

pip install --quiet --upgrade networkx plotly scikit-learn 2>&1 | grep -E "Successfully|already"

if [ $? -eq 0 ]; then
    echo "   ✅ Core dependencies installed"
else
    echo "   ❌ Installation failed"
    exit 1
fi
echo ""

# Install optional dependencies
echo "3️⃣ Installing optional visualization..."
pip install --quiet pyvis 2>&1 | grep -E "Successfully|already"

if [ $? -eq 0 ]; then
    echo "   ✅ PyVis installed (enhanced visualization)"
else
    echo "   ⚠️ PyVis installation failed (will use Plotly fallback)"
fi
echo ""

# Verify dependencies
echo "4️⃣ Verifying installation..."
python -c "
import sys

packages = {
    'networkx': 'Network analysis',
    'plotly': 'Visualization',
    'sklearn': 'ML utilities',
    'langchain': 'RAG framework',
    'faiss': 'Vector search',
    'sentence_transformers': 'Embeddings'
}

missing = []
for pkg, desc in packages.items():
    try:
        __import__(pkg)
        print(f'   ✅ {desc}')
    except ImportError:
        print(f'   ❌ {desc} ({pkg})')
        missing.append(pkg)

if missing:
    print(f'\n   Missing: {missing}')
    sys.exit(1)
else:
    print('\n   ✅ All core packages verified!')
" 2>&1

if [ $? -ne 0 ]; then
    echo "   ❌ Some packages are missing"
    echo "   Run: pip install -r requirements.txt"
    exit 1
fi
echo ""

# Run test suite
echo "5️⃣ Running test suite..."
echo ""

if [ -f "test_kg_rag_system.py" ]; then
    python test_kg_rag_system.py 2>&1 | tail -30
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "   ✅ All tests passed!"
    else
        echo ""
        echo "   ⚠️ Some tests failed (see above)"
    fi
else
    echo "   ⚠️ test_kg_rag_system.py not found"
    echo "   Skipping tests..."
fi
echo ""

# Check Ollama
echo "6️⃣ Checking Ollama (for RAG features)..."
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "   ✅ Ollama is running"
    
    models=$(curl -s http://localhost:11434/api/tags | grep -o '"name":"[^"]*"' | cut -d'"' -f4)
    if echo "$models" | grep -q "llama3"; then
        echo "   ✅ llama3 model available"
    else
        echo "   ⚠️ llama3 not found. Install: ollama pull llama3"
    fi
else
    echo "   ⚠️ Ollama not running"
    echo "   Start with: ollama serve"
    echo "   (Required for RAG comparison features)"
fi
echo ""

# Summary
echo "============================================================"
echo "📊 SETUP SUMMARY"
echo "============================================================"
echo ""
echo "✅ Installed:"
echo "   - NetworkX (graph construction)"
echo "   - Plotly (visualization)"
echo "   - Scikit-learn (similarity)"
echo ""

if pip show pyvis >/dev/null 2>&1; then
    echo "✅ Optional:"
    echo "   - PyVis (enhanced visualization)"
    echo ""
fi

echo "🎯 Next Steps:"
echo ""
echo "1. Start Athena:"
echo "   streamlit run app.py"
echo ""
echo "2. Upload a research paper"
echo ""
echo "3. Try new features:"
echo "   - 🕸️ Knowledge Graph tab: Visualize entities"
echo "   - 📚 Advanced RAG tab: Multi-document Q&A"
echo ""
echo "4. Read the guide:"
echo "   See KNOWLEDGE_GRAPH_RAG_GUIDE.md for details"
echo ""

echo "💡 Tips:"
echo "   - Use PyVis for interactive graphs"
echo "   - Add multiple papers for comparison"
echo "   - Combine KG + RAG for best results"
echo ""

echo "📚 Resources:"
echo "   - Test suite: python test_kg_rag_system.py"
echo "   - Examples: See guide documentation"
echo "   - Troubleshooting: Check guide FAQ section"
echo ""

echo "============================================================"
echo "🎉 Setup Complete!"
echo "============================================================"
echo ""