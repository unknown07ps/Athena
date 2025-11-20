
import streamlit as st
import requests
import time
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tracker_integration import get_tracker, get_calc

OLLAMA_URL = "http://localhost:11434/api/generate"


def make_qa_chain(pdf_text: str, chunk_size: int = 2000, k: int = 3, 
                  model: str = "llama3", track: bool = True):
    """
    Offline Q&A system with agent tracking.
    Returns a callable function that answers questions.
    """
    tracker = get_tracker()
    calc = get_calc()
    
    start = time.time()
    if track:
        tracker.log_action("build_qa_index",
                          chunk_size=chunk_size,
                          k=k,
                          text_length=len(pdf_text))
    
    try:
        #  Create vector embeddings
        embed = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        #  Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(pdf_text)
        
        if not texts:
            raise ValueError("No text chunks created from PDF")
        
        # Create FAISS index
        vectordb = FAISS.from_texts(texts, embed)
        retriever = vectordb.as_retriever(search_kwargs={"k": k})
        
        duration = time.time() - start
        print(f" QA Index created with {len(texts)} chunks in {duration:.2f}s")
        
        if track:
            tracker.add_reward(calc.task_completion(True),
                             f"QA index built ({len(texts)} chunks)")
            tracker.add_reward(calc.response_time(duration, 10.0),
                             f"Build time: {duration:.2f}s")
            tracker.add_reward(3, f"Retrieval depth: {k} documents")
        
    except Exception as e:
        print(f"❌ Error creating QA index: {e}")
        if track:
            tracker.add_reward(calc.error_penalty(),
                             f"QA index build failed: {str(e)}")
        raise

    def answer(question: str, track_answer: bool = True) -> str:
        """Answer questions based on the PDF content"""
        tracker = get_tracker()
        calc = get_calc()
        
        start = time.time()
        if track_answer:
            tracker.log_action("answer_question",
                              question=question[:50],
                              k=k)
        
        try:
            # Retrieve most relevant context
            try:
                docs = retriever.invoke(question)
            except AttributeError:
                docs = retriever.get_relevant_documents(question)
            
            retrieval_duration = time.time() - start
            
            if not docs:
                if track_answer:
                    tracker.add_reward(-2, "No relevant context found")
                return "⚠️ No relevant context found in the document."
            
            if track_answer:
                tracker.add_reward(2, f"Retrieved {len(docs)} context chunks")
            
            context = "\n\n---\n\n".join([doc.page_content for doc in docs])
            
            # Build prompt
            prompt = f"""You are Athena, an intelligent AI research assistant.
Answer the question based strictly on the provided context below.
If the context doesn't contain enough information, say: "I don't have enough information from this document to answer that question."

Context:
{context}

Question: {question}

Answer:"""
            
            # LOG ACTION: Call LLM
            llm_start = time.time()
            if track_answer:
                tracker.log_action("call_ollama_qa",
                                  model=model,
                                  prompt_length=len(prompt))
            
            # Send request to Ollama API
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 500
                }
            }
            
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            llm_duration = time.time() - llm_start
            
            if response.status_code != 200:
                if track_answer:
                    tracker.add_reward(calc.error_penalty(),
                                     f"Ollama error: {response.status_code}")
                return f"❌ Ollama API error {response.status_code}: {response.text}"
            
            data = response.json()
            answer_text = data.get("response", "").strip()
            
            if not answer_text:
                if track_answer:
                    tracker.add_reward(-3, "Empty answer from LLM")
                return " No answer received from the model."
            
            total_duration = time.time() - start
            
            # REWARDS
            if track_answer:
                tracker.add_reward(calc.task_completion(True),
                                 "Answer generated successfully")
                tracker.add_reward(calc.response_time(total_duration, 15.0),
                                 f"Total time: {total_duration:.2f}s")
                
                # Quality based on answer length
                if len(answer_text) > 100:
                    tracker.add_reward(5, "Detailed answer")
                elif len(answer_text) > 50:
                    tracker.add_reward(3, "Good answer")
                else:
                    tracker.add_reward(1, "Brief answer")
            
            return answer_text
            
        except requests.exceptions.Timeout:
            if track_answer:
                tracker.add_reward(calc.error_penalty(), "Request timeout")
            return "❌ Request timed out. The model might be processing a large context."
        except requests.exceptions.ConnectionError:
            if track_answer:
                tracker.add_reward(calc.error_penalty(), "Connection error")
            return "❌ Could not connect to Ollama. Make sure it's running."
        except Exception as e:
            if track_answer:
                tracker.add_reward(calc.error_penalty(), f"QA error: {str(e)}")
            return f"❌ Error during Q&A: {str(e)}"
    
    return answer