# document_comparison.py - Advanced Document Comparison Engine

import requests
from typing import Dict, List
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class DocumentComparison:
    """
    Advanced document comparison system using:
    - Semantic similarity (embeddings)
    - LLM-based analysis
    - Side-by-side comparison
    """
    
    def __init__(self, model="llama3"):
        self.model = model
        self.ollama_url = "http://localhost:11434/api/generate"
        self.documents = {}  # Store documents: {name: text}
        self.embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    def add_document(self, name: str, text: str):
        """Add a document to the comparison pool"""
        self.documents[name] = text
        print(f"✅ Added document: {name} ({len(text)} characters)")
    
    def clear_documents(self):
        """Clear all stored documents"""
        self.documents = {}
    
    def get_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts using embeddings
        Returns: similarity score (0-1)
        """
        try:
            # Get embeddings
            emb1 = self.embeddings_model.embed_query(text1[:5000])  # Limit for performance
            emb2 = self.embeddings_model.embed_query(text2[:5000])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            
            return float(similarity)
        except Exception as e:
            print(f"❌ Error calculating similarity: {e}")
            return 0.0
    
    def compare_documents(self, doc1_name: str, doc2_name: str) -> Dict:
        """
        Compare two documents and return comprehensive analysis
        
        Returns:
            {
                'summary': overall comparison summary,
                'similarities': common topics/themes,
                'differences': key differences,
                'recommendations': usage recommendations,
                'similarity_score': numerical similarity
            }
        """
        if doc1_name not in self.documents or doc2_name not in self.documents:
            raise ValueError("Documents not found. Please add them first.")
        
        text1 = self.documents[doc1_name]
        text2 = self.documents[doc2_name]
        
        print(f"📊 Comparing: {doc1_name} vs {doc2_name}")
        
        # Calculate semantic similarity
        similarity_score = self.get_semantic_similarity(text1, text2)
        print(f"   Similarity score: {similarity_score:.2%}")
        
        # Generate AI-powered comparison
        comparison = self._generate_comparison_analysis(
            doc1_name, text1, doc2_name, text2, similarity_score
        )
        
        return comparison
    
    def _generate_comparison_analysis(self, doc1_name: str, text1: str, 
                                     doc2_name: str, text2: str, 
                                     similarity_score: float) -> Dict:
        """
        Use LLM to generate detailed comparison analysis
        """
        try:
            # Prepare comparison prompt
            prompt = f"""You are an expert research analyst. Compare these two documents and provide a detailed analysis.

DOCUMENT 1: {doc1_name}
{text1[:3000]}
...

DOCUMENT 2: {doc2_name}
{text2[:3000]}
...

Semantic Similarity Score: {similarity_score:.2%}

Provide a comprehensive comparison with the following sections:

1. SUMMARY: Brief overview of what each document is about and their relationship

2. SIMILARITIES: Common topics, themes, methodologies, or findings (bullet points)

3. KEY DIFFERENCES: Major differences in approach, focus, conclusions, or scope (bullet points)

4. RECOMMENDATIONS: When to use each document, which is better for specific purposes

Keep your analysis professional, clear, and concise. Focus on substantive differences, not minor details.
"""
            
            # Call Ollama
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 1000
                }
            }
            
            response = requests.post(self.ollama_url, json=payload, timeout=180)
            
            if response.status_code != 200:
                return self._generate_fallback_comparison(similarity_score)
            
            data = response.json()
            full_analysis = data.get("response", "").strip()
            
            # Parse the response into sections
            sections = self._parse_comparison_response(full_analysis)
            sections['similarity_score'] = similarity_score
            
            return sections
            
        except requests.exceptions.ConnectionError:
            print("❌ Could not connect to Ollama")
            return self._generate_fallback_comparison(similarity_score)
        except Exception as e:
            print(f"❌ Error generating comparison: {e}")
            return self._generate_fallback_comparison(similarity_score)
    
    def _parse_comparison_response(self, response: str) -> Dict:
        """
        Parse the LLM response into structured sections
        """
        sections = {
            'summary': '',
            'similarities': '',
            'differences': '',
            'recommendations': ''
        }
        
        # Try to extract sections
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line_lower = line.lower().strip()
            
            if 'summary' in line_lower and ':' in line:
                current_section = 'summary'
                continue
            elif 'similarit' in line_lower and ':' in line:
                current_section = 'similarities'
                continue
            elif 'difference' in line_lower and ':' in line:
                current_section = 'differences'
                continue
            elif 'recommendation' in line_lower and ':' in line:
                current_section = 'recommendations'
                continue
            
            if current_section and line.strip():
                sections[current_section] += line + '\n'
        
        # Clean up sections
        for key in sections:
            sections[key] = sections[key].strip()
            if not sections[key]:
                sections[key] = f"No {key} information available."
        
        return sections
    
    def _generate_fallback_comparison(self, similarity_score: float) -> Dict:
        """
        Generate a basic comparison when LLM is unavailable
        """
        similarity_interpretation = "very similar" if similarity_score > 0.7 else \
                                   "moderately similar" if similarity_score > 0.4 else \
                                   "quite different"
        
        return {
            'summary': f"Based on semantic analysis, these documents are {similarity_interpretation} (similarity: {similarity_score:.2%}). For a detailed comparison, please ensure Ollama is running.",
            'similarities': "Semantic analysis indicates some overlap in topics and terminology.",
            'differences': "Detailed difference analysis requires LLM processing.",
            'recommendations': "Upload documents again after starting Ollama for full analysis.",
            'similarity_score': similarity_score
        }
    
    def batch_compare(self, doc_names: List[str]) -> Dict:
        """
        Compare multiple documents at once
        Returns: similarity matrix and analysis
        """
        n = len(doc_names)
        similarity_matrix = np.zeros((n, n))
        
        for i, doc1 in enumerate(doc_names):
            for j, doc2 in enumerate(doc_names):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                elif i < j:
                    sim = self.get_semantic_similarity(
                        self.documents[doc1],
                        self.documents[doc2]
                    )
                    similarity_matrix[i][j] = sim
                    similarity_matrix[j][i] = sim
        
        return {
            'documents': doc_names,
            'similarity_matrix': similarity_matrix.tolist(),
            'most_similar': self._find_most_similar_pair(doc_names, similarity_matrix),
            'least_similar': self._find_least_similar_pair(doc_names, similarity_matrix)
        }
    
    def _find_most_similar_pair(self, docs: List[str], matrix: np.ndarray):
        """Find the most similar pair of documents"""
        n = len(docs)
        max_sim = 0
        pair = None
        
        for i in range(n):
            for j in range(i+1, n):
                if matrix[i][j] > max_sim:
                    max_sim = matrix[i][j]
                    pair = (docs[i], docs[j])
        
        return {'pair': pair, 'similarity': float(max_sim)}
    
    def _find_least_similar_pair(self, docs: List[str], matrix: np.ndarray):
        """Find the least similar pair of documents"""
        n = len(docs)
        min_sim = 1.0
        pair = None
        
        for i in range(n):
            for j in range(i+1, n):
                if matrix[i][j] < min_sim:
                    min_sim = matrix[i][j]
                    pair = (docs[i], docs[j])
        
        return {'pair': pair, 'similarity': float(min_sim)}


# Test function
if __name__ == "__main__":
    print("🧪 Testing Document Comparison Engine\n")
    
    comparator = DocumentComparison()
    
    # Sample documents
    doc1 = """
    This paper introduces a novel transformer architecture for natural language processing.
    The model uses multi-head attention mechanisms and achieves state-of-the-art results
    on multiple benchmarks including GLUE and SQuAD. We trained on 100GB of text data
    using 8 GPUs for 2 weeks.
    """
    
    doc2 = """
    We present a new convolutional neural network for computer vision tasks.
    Our architecture uses residual connections and batch normalization to achieve
    excellent performance on ImageNet classification. Training was done on 4 GPUs
    for 3 days using the ImageNet dataset.
    """
    
    doc3 = """
    This research explores transformer models in natural language understanding.
    We implement self-attention layers with improved efficiency and test on
    various NLP benchmarks. The model was trained using modern GPU infrastructure
    on large text corpora.
    """
    
    # Add documents
    comparator.add_document("NLP_Transformer.pdf", doc1)
    comparator.add_document("CV_CNN.pdf", doc2)
    comparator.add_document("NLP_Efficient.pdf", doc3)
    
    # Compare two documents
    print("\n📊 Comparing NLP_Transformer vs CV_CNN:")
    result = comparator.compare_documents("NLP_Transformer.pdf", "CV_CNN.pdf")
    print(f"Similarity: {result['similarity_score']:.2%}")
    print(f"\nSummary:\n{result['summary'][:200]}...")
    
    # Batch comparison
    print("\n\n📊 Batch comparison of all documents:")
    batch_result = comparator.batch_compare([
        "NLP_Transformer.pdf",
        "CV_CNN.pdf", 
        "NLP_Efficient.pdf"
    ])
    
    print(f"\nMost similar: {batch_result['most_similar']['pair']}")
    print(f"Similarity: {batch_result['most_similar']['similarity']:.2%}")
    
    print(f"\nLeast similar: {batch_result['least_similar']['pair']}")
    print(f"Similarity: {batch_result['least_similar']['similarity']:.2%}")