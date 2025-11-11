# chat_engine.py - Conversational AI with Memory for Athena

import requests
from datetime import datetime


class AthenaChat:
    """Conversational AI with memory for research discussions"""
    
    def __init__(self, model="llama3", temperature=0.3):
        self.model = model
        self.temperature = temperature
        self.chat_history = []
        self.ollama_url = "http://localhost:11434/api/generate"
    
    def chat(self, user_message):
        """
        Send a message and get a response with conversation context
        
        Args:
            user_message: User's message
            
        Returns:
            Athena's response
        """
        try:
            # Build conversation context
            context = self._build_context()
            
            # Create prompt with history
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
            return "❌ Could not connect to Ollama. Make sure it's running."
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def _build_context(self):
        """Build conversation context from history"""
        if not self.chat_history:
            return "This is the start of the conversation."
        
        # Include last 5 exchanges
        recent = self.chat_history[-5:]
        context_lines = ["Previous conversation:"]
        
        for exchange in recent:
            context_lines.append(f"User: {exchange['user']}")
            context_lines.append(f"Athena: {exchange['assistant']}")
        
        return "\n".join(context_lines)
    
    def clear_history(self):
        """Clear conversation history"""
        self.chat_history = []
    
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
    print("🧠 Testing Athena Chat Engine\n")
    
    chat = AthenaChat()
    
    # Test conversation
    test_messages = [
        "Hello! Can you explain what transformers are in AI?",
        "What are some applications of transformers?",
        "Which is better - CNN or Transformer for image tasks?"
    ]
    
    for msg in test_messages:
        print(f"User: {msg}")
        response = chat.chat(msg)
        print(f"Athena: {response}\n")
        print("-" * 60 + "\n")
    
    # Show history
    print("\n📝 Chat History:")
    history = chat.get_history()
    print(f"Total exchanges: {len(history)}")
    
    # Export
    filename = chat.export_history()
    print(f"\n✅ Exported to: {filename}")