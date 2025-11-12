#!/usr/bin/env python3
"""
Complete test suite for Knowledge Graph + Advanced RAG system
Run this to verify all components work correctly
"""

import sys
import os


def test_imports():
    """Test if all required packages are installed"""
    print("=" * 70)
    print("📦 TEST 1: Package Imports")
    print("=" * 70)
    
    packages = {
        'networkx': 'networkx',
        'plotly': 'plotly',
        'sklearn': 'scikit-learn',
        'langchain': 'langchain',
        'faiss': 'faiss-cpu',
        'sentence_transformers': 'sentence-transformers'
    }
    
    missing = []
    for module, package in packages.items():
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Install: pip install {package}")
            missing.append(package)
    
    # Optional packages
    print("\nOptional:")
    try:
        import pyvis
        print("✅ pyvis (recommended for visualization)")
    except ImportError:
        print("⚠️  pyvis not installed - using Plotly fallback")
    
    if missing:
        print(f"\n❌ Missing required packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("\n✅ All required packages installed!")
    return True


def test_knowledge_graph():
    """Test knowledge graph construction"""
    print("\n" + "=" * 70)
    print("🕸️ TEST 2: Knowledge Graph Construction")
    print("=" * 70)
    
    try:
        from knowledge_graph import KnowledgeGraphBuilder
        
        kg = KnowledgeGraphBuilder()
        print("✅ Knowledge graph builder initialized")
        
        # Sample text
        sample_text = """
        We propose the Transformer architecture using self-attention mechanisms.
        The model is evaluated on WMT 2014 dataset achieving 28.4 BLEU score.
        Our approach outperforms LSTM models on machine translation tasks.
        We use Adam optimizer for training on ImageNet dataset.
        """
        
        print("\n📊 Building graph from sample text...")
        graph = kg.build_graph(sample_text, "Test Paper")
        
        summary = kg.get_graph_summary()
        
        print(f"\n✅ Graph built successfully:")
        print(f"   Nodes: {summary['total_nodes']}")
        print(f"   Edges: {summary['total_edges']}")
        print(f"   Node types: {list(summary['node_types'].keys())}")
        
        # Test queries
        print("\n🔍 Testing graph queries...")
        results = kg.query_graph("transformer")
        print(f"   Query 'transformer': {len(results)} results")
        
        # Test export
        print("\n📤 Testing export...")
        export_data = kg.export_to_cytoscape()
        print(f"   Exported {len(export_data['elements'])} elements")
        
        print("\n✅ Knowledge graph tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Knowledge graph test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_advanced_rag():
    """Test advanced RAG system"""
    print("\n" + "=" * 70)
    print("📚 TEST 3: Advanced RAG System")
    print("=" * 70)
    
    try:
        from advanced_rag import AdvancedRAG
        
        rag = AdvancedRAG(chunk_size=500)
        print("✅ Advanced RAG initialized")
        
        # Add sample documents
        doc1 = """
        Transformers use self-attention mechanisms to process sequences.
        They achieve state-of-the-art results on machine translation.
        The architecture consists of encoder and decoder layers.
        """
        
        doc2 = """
        BERT uses bidirectional transformers for language understanding.
        It is pre-trained on masked language modeling tasks.
        BERT achieves excellent results on question answering.
        """
        
        print("\n📚 Adding documents...")
        rag.add_document("transformer", "Transformer Paper", doc1)
        rag.add_document("bert", "BERT Paper", doc2)
        
        summary = rag.get_document_summary()
        print(f"✅ Added {summary['total_documents']} documents")
        
        # Test retrieval
        print("\n🔍 Testing context retrieval...")
        contexts = rag.retrieve_context("attention mechanisms", k=3)
        print(f"   Retrieved {len(contexts)} contexts")
        
        if contexts:
            print(f"   Top similarity: {contexts[0][2]:.2%}")
        
        # Test comparison (requires Ollama)
        print("\n🔬 Testing document comparison...")
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                print("   Ollama detected, testing comparison...")
                result = rag.compare_documents(
                    "attention",
                    doc_ids=["transformer", "bert"],
                    k=2
                )
                
                if 'comparison' in result:
                    print(f"   ✅ Comparison generated ({len(result['comparison'])} chars)")
                else:
                    print(f"   ⚠️ Comparison result: {result}")
            else:
                print("   ⚠️ Ollama not running, skipping LLM tests")
        except:
            print("   ⚠️ Ollama not available, skipping LLM tests")
        
        print("\n✅ Advanced RAG tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Advanced RAG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization():
    """Test visualization components"""
    print("\n" + "=" * 70)
    print("🎨 TEST 4: Visualization")
    print("=" * 70)
    
    try:
        from knowledge_graph import KnowledgeGraphBuilder
        
        kg = KnowledgeGraphBuilder()
        
        sample_text = """
        Transformers use attention. BERT uses transformers.
        GPT-3 improves BERT. All evaluated on SQUAD dataset.
        """
        
        kg.build_graph(sample_text, "Test")
        
        # Test Plotly (always available)
        print("\n📊 Testing Plotly visualization...")
        try:
            import plotly.graph_objects as go
            import networkx as nx
            
            pos = nx.spring_layout(kg.graph)
            print("✅ Plotly visualization ready")
        except ImportError:
            print("❌ Plotly not available")
            return False
        
        # Test PyVis (optional)
        print("\n🎨 Testing PyVis visualization...")
        try:
            from pyvis.network import Network
            net = Network(height="400px", width="100%")
            
            for node in list(kg.graph.nodes())[:5]:
                net.add_node(node)
            
            print("✅ PyVis available (recommended)")
        except ImportError:
            print("⚠️ PyVis not installed (optional)")
            print("   Install for better UX: pip install pyvis")
        
        print("\n✅ Visualization tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration between components"""
    print("\n" + "=" * 70)
    print("🔗 TEST 5: Integration")
    print("=" * 70)
    
    try:
        from knowledge_graph import KnowledgeGraphBuilder
        from advanced_rag import AdvancedRAG
        
        # Sample paper
        paper_text = """
        Title: Attention Mechanisms in Neural Networks
        
        We propose using self-attention for sequence processing.
        The method is evaluated on WMT 2014 dataset achieving 
        28.4 BLEU score. Our transformer architecture outperforms
        LSTM models on machine translation tasks.
        
        We use multi-head attention with 8 heads and 512-dimensional
        embeddings. Training is done with Adam optimizer on ImageNet.
        Results show 92% accuracy on image classification.
        """
        
        print("\n1️⃣ Building knowledge graph...")
        kg = KnowledgeGraphBuilder()
        graph = kg.build_graph(paper_text, "Attention Paper")
        
        entities = kg.extract_entities(paper_text)
        print(f"   Extracted {sum(len(v) for v in entities.values())} entities")
        
        print("\n2️⃣ Setting up RAG system...")
        rag = AdvancedRAG()
        rag.add_document("attention_paper", "Attention Paper", paper_text)
        
        print("\n3️⃣ Testing combined workflow...")
        
        # Find entities in graph
        query_results = kg.query_graph("attention", k=3)
        print(f"   Graph query: {len(query_results)} results")
        
        # Retrieve context with RAG
        contexts = rag.retrieve_context("attention mechanisms", k=3)
        print(f"   RAG retrieval: {len(contexts)} contexts")
        
        # Verify overlap
        if query_results and contexts:
            print("   ✅ Both systems working together")
        
        print("\n✅ Integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run complete test suite"""
    print("\n")
    print("=" * 70)
    print("🧠 ATHENA KNOWLEDGE GRAPH + RAG TEST SUITE")
    print("=" * 70)
    print("\nThis will test all components of the KG + RAG system\n")
    
    results = []
    
    # Run tests
    results.append(("Package Imports", test_imports()))
    
    if results[-1][1]:  # Only continue if imports work
        results.append(("Knowledge Graph", test_knowledge_graph()))
        results.append(("Advanced RAG", test_advanced_rag()))
        results.append(("Visualization", test_visualization()))
        results.append(("Integration", test_integration()))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("💡 NEXT STEPS")
    print("=" * 70)
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! You're ready to use KG + RAG!")
        print("\n✅ What you can do now:")
        print("   1. Start Athena: streamlit run app.py")
        print("   2. Upload a research paper")
        print("   3. Build knowledge graph (🕸️ tab)")
        print("   4. Add to RAG system (📚 tab)")
        print("   5. Ask contextual questions")
        print("   6. Compare multiple papers")
        
    elif passed_count >= 3:
        print("\n✅ Core features working!")
        print("\n⚠️ Some optional features failed:")
        print("   - Check Ollama for RAG comparisons")
        print("   - Install PyVis for better visualization")
        print("\n💡 You can still use the system with current features")
        
    else:
        print("\n❌ Critical tests failed!")
        print("\n🔧 Troubleshooting:")
        print("   1. Install missing packages:")
        print("      pip install networkx plotly scikit-learn")
        print("   2. Verify LangChain installation:")
        print("      pip install langchain faiss-cpu sentence-transformers")
        print("   3. Check Python version (3.8+ required):")
        print(f"      Current: {sys.version}")
        print("   4. Re-run this test after fixing issues")
    
    print("\n" + "=" * 70)
    print("📚 Documentation: See KNOWLEDGE_GRAPH_RAG_GUIDE.md")
    print("=" * 70)
    
    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)