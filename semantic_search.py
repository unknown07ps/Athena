from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tracker_integration import get_tracker, get_calc
import time


def build_semantic_index(pdf_text: str, chunk_size: int = 300, 
                        chunk_overlap: int = 50, track: bool = True):
    """
    Build a FAISS semantic index with agent tracking.
    """
    tracker = get_tracker()
    calc = get_calc()
    
    start = time.time()
    if track:
        tracker.log_action("build_semantic_index",
                          chunk_size=chunk_size,
                          chunk_overlap=chunk_overlap,
                          text_length=len(pdf_text))
    
    try:
        print(f" Building index with chunk_size={chunk_size}, overlap={chunk_overlap}")
        print(f" Input text length: {len(pdf_text)} characters")
        
        # Initialize embedding model
        embed_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        print(" Embedding model loaded")
        
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
        texts = text_splitter.split_text(pdf_text)
        
        # Clean up chunks
        texts = [' '.join(text.split()) for text in texts if text.strip()]
        
        print(f" Split into {len(texts)} chunks")
        
        if not texts:
            raise ValueError("No text chunks created from PDF")
        
        # Create FAISS index
        vectordb = FAISS.from_texts(texts, embed_model)
        
        duration = time.time() - start
        print(f" Semantic index created with {len(texts)} chunks in {duration:.2f}s")
        
        if track:
            tracker.add_reward(calc.task_completion(True),
                             f"Semantic index built ({len(texts)} chunks)")
            tracker.add_reward(calc.response_time(duration, 10.0),
                             f"Build time: {duration:.2f}s")
        
        return vectordb
        
    except Exception as e:
        print(f"❌ Error building semantic index: {e}")
        if track:
            tracker.add_reward(calc.error_penalty(),
                             f"Index build failed: {str(e)}")
        raise


def search_semantic(vectordb, query: str, k: int = 10, track: bool = True):
    """
    Perform semantic search with agent tracking.
    Returns results as [(text, similarity_score), ...]
    """
    tracker = get_tracker()
    calc = get_calc()
    
    start = time.time()
    if track:
        tracker.log_action("semantic_search",
                          query=query[:50],
                          k=k)
    
    try:
        print(f"\n Searching for: '{query}' (k={k})")
        
        if not query or not query.strip():
            print(" Empty query provided")
            if track:
                tracker.add_reward(-1, "Empty search query")
            return []
        
        # Perform similarity search with scores
        results = vectordb.similarity_search_with_score(query, k=k)
        print(f" FAISS returned {len(results)} results")
        
        if not results:
            print(" No results from FAISS")
            if track:
                tracker.add_reward(-2, "No search results")
            return []
        
        # Convert to similarity scores and format
        formatted_results = []
        for i, (doc, distance) in enumerate(results):
            similarity = 1 / (1 + distance)
            clean_text = ' '.join(doc.page_content.split())
            formatted_results.append((clean_text, similarity))
        
        # Sort by similarity
        formatted_results.sort(key=lambda x: x[1], reverse=True)
        
        duration = time.time() - start
        
        # REWARDS
        if track:
            tracker.add_reward(calc.task_completion(True),
                             f"Found {len(results)} matches")
            tracker.add_reward(calc.response_time(duration, 3.0),
                             f"Search time: {duration:.2f}s")
            
            # Quality based on top result similarity
            top_similarity = formatted_results[0][1]
            tracker.add_reward(calc.quality_score(top_similarity),
                             f"Top match: {top_similarity:.0%}")
        
        print(f" Returning {len(formatted_results)} results")
        return formatted_results
        
    except Exception as e:
        print(f"❌ Error during semantic search: {e}")
        if track:
            tracker.add_reward(calc.error_penalty(),
                             f"Search error: {str(e)}")
        raise


def search_semantic_with_threshold(vectordb, query: str, k: int = 10, 
                                  min_similarity: float = 0.3, track: bool = True):
    """
    Perform semantic search with a minimum similarity threshold.
    """
    all_results = search_semantic(vectordb, query, k=k, track=track)
    filtered_results = [(text, score) for text, score in all_results 
                       if score >= min_similarity]
    
    print(f" Filtered: {len(filtered_results)}/{len(all_results)} results "
          f"above similarity {min_similarity}")
    
    if track and filtered_results:
        tracker = get_tracker()
        tracker.add_reward(2, f"Filtered to {len(filtered_results)} high-quality results")
    
    return filtered_results