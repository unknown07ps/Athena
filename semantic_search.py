# semantic_search.py — Improved with better text chunking and display
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np


def build_semantic_index(pdf_text: str, chunk_size: int = 300, chunk_overlap: int = 50):
    """
    Build a FAISS semantic index with smaller, more readable chunks.
    Returns the vectordb for searching.
    """
    try:
        print(f"🔧 Building index with chunk_size={chunk_size}, overlap={chunk_overlap}")
        print(f"📄 Input text length: {len(pdf_text)} characters")
        
        # Initialize embedding model
        embed_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        print("✅ Embedding model loaded")
        
        # Split text with better separators for resumes/papers
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]  # Better separation
        )
        texts = text_splitter.split_text(pdf_text)
        
        # Clean up chunks - remove excessive whitespace
        texts = [' '.join(text.split()) for text in texts if text.strip()]
        
        print(f"✂️ Split into {len(texts)} chunks")
        if texts:
            print(f"🔍 Sample chunk: {texts[0][:100]}...")
        
        if not texts:
            raise ValueError("No text chunks created from PDF")
        
        # Create FAISS index
        vectordb = FAISS.from_texts(texts, embed_model)
        
        print(f"✅ Semantic index created with {len(texts)} chunks")
        return vectordb
        
    except Exception as e:
        print(f"❌ Error building semantic index: {e}")
        raise


def search_semantic(vectordb, query: str, k: int = 10):
    """
    Perform semantic search with better formatting.
    Returns results as [(text, similarity_score), ...]
    """
    try:
        print(f"\n🔍 Searching for: '{query}' (k={k})")
        
        if not query or not query.strip():
            print("⚠️ Empty query provided")
            return []
        
        # Perform similarity search with scores
        results = vectordb.similarity_search_with_score(query, k=k)
        print(f"📊 FAISS returned {len(results)} results")
        
        if not results:
            print("⚠️ No results from FAISS")
            return []
        
        # Convert to similarity scores and format
        formatted_results = []
        for i, (doc, distance) in enumerate(results):
            # Convert distance to similarity (0-1 range)
            similarity = 1 / (1 + distance)
            
            # Clean the text - remove extra whitespace
            clean_text = ' '.join(doc.page_content.split())
            
            print(f"   Result {i+1}: distance={distance:.4f}, similarity={similarity:.4f}")
            formatted_results.append((clean_text, similarity))
        
        # Sort by similarity (highest first)
        formatted_results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"✅ Returning {len(formatted_results)} results")
        return formatted_results
        
    except Exception as e:
        print(f"❌ Error during semantic search: {e}")
        import traceback
        traceback.print_exc()
        raise


def search_semantic_with_threshold(vectordb, query: str, k: int = 10, min_similarity: float = 0.3):
    """
    Perform semantic search with a minimum similarity threshold.
    """
    all_results = search_semantic(vectordb, query, k=k)
    filtered_results = [(text, score) for text, score in all_results if score >= min_similarity]
    
    print(f"🔍 Filtered: {len(filtered_results)}/{len(all_results)} results above similarity {min_similarity}")
    
    return filtered_results