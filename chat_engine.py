# chat_engine.py - Context-aware conversational AI for Athena

import requests
from datetime import datetime


class AthenaChat:
    """Conversational AI with memory and PDF context awareness"""
    
    def __init__(self, model="llama3", temperature=0.3):
        self.model = model
        self.temperature = temperature
        self.chat_history = []
        self.ollama_url = "http://localhost:11434/api/generate"
        self.pdf_context = None  # Store PDF content for context
    
    def set_pdf_context(self, pdf_text: str):
        """
        Set the PDF context for the chat session.
        This allows Athena to answer questions based on the uploaded document.
        """
        self.pdf_context = pdf_text
        print(f"✅ PDF context set ({len(pdf_text)} characters)")
    
    def chat(self, user_message: str):
        """
        Send a message and get a response with conversation and PDF context
        
        Args:
            user_message: User's message
            
        Returns:
            Athena's response
        """
        try:
            # Build conversation context
            context = self._build_context()
            
            # Create prompt with history and PDF context
            if self.pdf_context:
                prompt = f"""You are Athena, an AI research assistant. You have access to the user's uploaded document.

IMPORTANT: When answering questions about the document, ONLY use information from the DOCUMENT CONTENT below. 
Do NOT make up or hallucinate information. If the document doesn't contain the answer, say so clearly.

DOCUMENT CONTENT:
{self.pdf_context[:3000]}

{context}

User: {user_message}
Athena:"""
            else:
                prompt = f"""You are Athena, an AI research assistant. You're knowledgeable, helpful, and professional.

{context}

User: {user_message}
Athena:"""
            
            # Call Ollama API
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": 500
                }
            }
            
            response = requests.post(self.ollama_url, json=payload, timeout=120)
            
            if response.status_code != 200:
                return f"❌ Error: {response.status_code}"
            
            data = response.json()
            assistant_message = data.get("response", "").strip()
            
            # Save to history
            self.chat_history.append({
                'timestamp': datetime.now(),
                'user': user_message,
                'assistant': assistant_message
            })
            
            return assistant_message
            
        except requests.exceptions.ConnectionError:
            return "❌ Could not connect to Ollama. Make sure it's running: `ollama serve`"
        except requests.exceptions.Timeout:
            return "❌ Request timed out. The model is taking too long to respond."
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def _build_context(self):
        """Build conversation context from history"""
        if not self.chat_history:
            return "This is the start of the conversation."
        
        # Include last 3 exchanges to keep context manageable
        recent = self.chat_history[-3:]
        context_lines = ["Previous conversation:"]
        
        for exchange in recent:
            context_lines.append(f"User: {exchange['user']}")
            context_lines.append(f"Athena: {exchange['assistant']}")
        
        return "\n".join(context_lines)
    
    def clear_history(self):
        """Clear conversation history"""
        self.chat_history = []
    
    def clear_pdf_context(self):
        """Clear PDF context"""
        self.pdf_context = None
    
    def get_history(self):
        """Get full conversation history"""
        return self.chat_history
    
    def export_history(self, filename="chat_history.txt"):
        """Export chat history to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("Athena Chat History\n")
            f.write("=" * 50 + "\n\n")
            
            for i, exchange in enumerate(self.chat_history, 1):
                f.write(f"Exchange {i}\n")
                f.write(f"Time: {exchange['timestamp']}\n")
                f.write(f"User: {exchange['user']}\n")
                f.write(f"Athena: {exchange['assistant']}\n")
                f.write("-" * 50 + "\n\n")
        
        return filename


# Test function
if __name__ == "__main__":
    print("🧠 Testing Athena Chat Engine with PDF Context\n")
    
    chat = AthenaChat()
    
    # Simulate PDF context
    sample_pdf = """
    SAGAR PRAJAPATI
    Email: sagar@example.com
    
    EXPERIENCE:
    AI Research Intern at COEP (May 2025 - Aug 2025)
    - Worked on computer vision projects
    - Developed CNN-based exam proctoring system
    
    EDUCATION:
    B.Tech Computer Engineering, COEP
    
    SKILLS:
    Python, Java, Machine Learning, Deep Learning
    
    ACHIEVEMENTS:
    - Solved 540+ DSA problems on GeeksforGeeks
    - Rank 234 on GeeksforGeeks
    """
    
    chat.set_pdf_context(sample_pdf)
    
    # Test conversation
    test_messages = [
        "What work experience does this candidate have?",
        "How many DSA problems have they solved?",
        "What are their technical skills?"
    ]
    
    for msg in test_messages:
        print(f"User: {msg}")
        response = chat.chat(msg)
        print(f"Athena: {response}\n")
        print("-" * 60 + "\n")