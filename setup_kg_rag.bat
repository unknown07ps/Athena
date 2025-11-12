@echo off
REM setup_kg_rag.bat - Windows setup for Knowledge Graph + RAG

echo ============================================================
echo 🧠 ATHENA KNOWLEDGE GRAPH + RAG SETUP (Windows)
echo ============================================================
echo.

REM Check Python
echo 1️⃣ Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo    ❌ Python not found! Install Python 3.8+
    echo    Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo    Found: Python %PYTHON_VERSION%
echo    ✅ Python is installed
echo.

REM Install core dependencies
echo 2️⃣ Installing core dependencies...
echo    This may take 2-3 minutes...
echo.

pip install --quiet --upgrade networkx plotly scikit-learn
if errorlevel 1 (
    echo    ❌ Installation failed!
    pause
    exit /b 1
)

echo    ✅ Core dependencies installed
echo.

REM Install optional
echo 3️⃣ Installing optional visualization...
pip install --quiet pyvis
if errorlevel 1 (
    echo    ⚠️ PyVis installation failed (will use Plotly)
) else (
    echo    ✅ PyVis installed (enhanced visualization)
)
echo.

REM Verify
echo 4️⃣ Verifying installation...
python -c "import networkx, plotly, sklearn, langchain, faiss, sentence_transformers; print('   ✅ All core packages verified!')"
if errorlevel 1 (
    echo    ❌ Some packages missing
    echo    Run: pip install -r requirements.txt
    pause
    exit /b 1
)
echo.

REM Test suite
echo 5️⃣ Running test suite...
echo.

if exist test_kg_rag_system.py (
    python test_kg_rag_system.py
    if errorlevel 1 (
        echo.
        echo    ⚠️ Some tests failed (see above)
    ) else (
        echo.
        echo    ✅ All tests passed!
    )
) else (
    echo    ⚠️ test_kg_rag_system.py not found
    echo    Skipping tests...
)
echo.

REM Check Ollama
echo 6️⃣ Checking Ollama (for RAG features)...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo    ⚠️ Ollama not running
    echo    Start with: ollama serve
    echo    (Required for RAG comparison features)
) else (
    echo    ✅ Ollama is running
)
echo.

REM Summary
echo ============================================================
echo 📊 SETUP SUMMARY
echo ============================================================
echo.
echo ✅ Installed:
echo    - NetworkX (graph construction)
echo    - Plotly (visualization)
echo    - Scikit-learn (similarity)
echo.

pip show pyvis >nul 2>&1
if not errorlevel 1 (
    echo ✅ Optional:
    echo    - PyVis (enhanced visualization)
    echo.
)

echo 🎯 Next Steps:
echo.
echo 1. Start Athena:
echo    streamlit run app.py
echo.
echo 2. Upload a research paper
echo.
echo 3. Try new features:
echo    - 🕸️ Knowledge Graph tab: Visualize entities
echo    - 📚 Advanced RAG tab: Multi-document Q&A
echo.
echo 4. Read the guide:
echo    See KNOWLEDGE_GRAPH_RAG_GUIDE.md for details
echo.

echo 💡 Tips:
echo    - Use PyVis for interactive graphs
echo    - Add multiple papers for comparison
echo    - Combine KG + RAG for best results
echo.

echo 📚 Resources:
echo    - Test suite: python test_kg_rag_system.py
echo    - Examples: See guide documentation
echo    - Troubleshooting: Check guide FAQ section
echo.

echo ============================================================
echo 🎉 Setup Complete!
echo ============================================================
echo.

pause