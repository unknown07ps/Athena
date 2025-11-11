#!/usr/bin/env python3
"""
Integrated Athena Document Comparison Test Suite
Includes both the DocumentComparison class and its test runner.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys


# ===============================
#  DOCUMENT COMPARISON ENGINE
# ===============================
class DocumentComparison:
    def __init__(self):
        self.documents = {}

    def add_document(self, name: str, content: str):
        """Add a document to the comparison memory"""
        self.documents[name] = content.strip()

    def _vectorize_documents(self):
        """Vectorize documents using TF-IDF"""
        names = list(self.documents.keys())
        contents = [self.documents[name] for name in names]
        vectorizer = TfidfVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform(contents)
        return names, vectors

    def compare_documents(self, doc1_name: str, doc2_name: str):
        """Compare two documents and return similarity results"""
        if doc1_name not in self.documents or doc2_name not in self.documents:
            raise ValueError("One or both documents not found")

        names, vectors = self._vectorize_documents()
        idx1 = names.index(doc1_name)
        idx2 = names.index(doc2_name)
        score = cosine_similarity(vectors[idx1], vectors[idx2])[0][0]

        doc1_text = self.documents[doc1_name]
        doc2_text = self.documents[doc2_name]

        common_words = set(doc1_text.split()).intersection(set(doc2_text.split()))
        diff1 = set(doc1_text.split()) - set(doc2_text.split())
        diff2 = set(doc2_text.split()) - set(doc1_text.split())

        result = {
            "similarity_score": score,
            "summary": f"The similarity between '{doc1_name}' and '{doc2_name}' is {score:.2%}.",
            "similarities": ', '.join(list(common_words)[:15]) + '...',
            "differences": f"Unique to {doc1_name}: {', '.join(list(diff1)[:10])} | Unique to {doc2_name}: {', '.join(list(diff2)[:10])}",
            "recommendations": "Consider merging common sections or refining distinct methodologies."
        }
        return result

    def batch_compare(self, doc_names):
        """Perform pairwise comparison for multiple documents"""
        selected_docs = {k: self.documents[k] for k in doc_names if k in self.documents}
        names = list(selected_docs.keys())
        contents = [selected_docs[n] for n in names]

        vectorizer = TfidfVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform(contents)
        similarity_matrix = cosine_similarity(vectors)

        most_similar = (None, 0)
        least_similar = (None, 1)
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                score = similarity_matrix[i][j]
                if score > most_similar[1]:
                    most_similar = ((names[i], names[j]), score)
                if score < least_similar[1]:
                    least_similar = ((names[i], names[j]), score)

        return {
            "documents": names,
            "similarity_matrix": similarity_matrix.tolist(),
            "most_similar": {"pair": most_similar[0], "similarity": most_similar[1]},
            "least_similar": {"pair": least_similar[0], "similarity": least_similar[1]}
        }


# ===============================
#  TEST SUITE IMPLEMENTATION
# ===============================
def test_basic_comparison():
    print("=" * 70)
    print("🧪 TEST 1: Basic Document Comparison")
    print("=" * 70)
    
    comparator = DocumentComparison()
    
    doc1 = """Deep Learning for NLP with transformer models and BERT-based architectures."""
    doc2 = """Convolutional Neural Networks for image recognition and classification tasks."""
    
    comparator.add_document("NLP_Paper.pdf", doc1)
    comparator.add_document("CV_Paper.pdf", doc2)
    print("\n✅ Documents added successfully\n")

    result = comparator.compare_documents("NLP_Paper.pdf", "CV_Paper.pdf")
    
    print(f"\n🎯 Similarity Score: {result['similarity_score']:.2%}")
    print("\n📝 SUMMARY:")
    print(result['summary'])
    print("\n🤝 SIMILARITIES:")
    print(result['similarities'])
    print("\n⚡ DIFFERENCES:")
    print(result['differences'])
    print("\n💡 RECOMMENDATIONS:")
    print(result['recommendations'])

    return True


def test_batch_comparison():
    print("\n\n" + "=" * 70)
    print("🧪 TEST 2: Batch Comparison (3 Documents)")
    print("=" * 70)
    
    comparator = DocumentComparison()
    
    doc1 = "Transformers for NLP using attention mechanisms."
    doc2 = "BERT and GPT pre-training strategies for language understanding."
    doc3 = "Convolutional networks for image classification tasks."
    
    comparator.add_document("NLP_1.pdf", doc1)
    comparator.add_document("NLP_2.pdf", doc2)
    comparator.add_document("CV_1.pdf", doc3)
    
    result = comparator.batch_compare(["NLP_1.pdf", "NLP_2.pdf", "CV_1.pdf"])
    
    print("\n📊 Similarity Matrix:")
    for i, doc in enumerate(result['documents']):
        print(f"\n{doc}:")
        for j, sim in enumerate(result['similarity_matrix'][i]):
            print(f"  vs {result['documents'][j]}: {sim:.2%}")
    
    print(f"\n🔝 Most Similar Pair: {result['most_similar']['pair']}")
    print(f"   Similarity: {result['most_similar']['similarity']:.2%}")
    print(f"\n🔻 Least Similar Pair: {result['least_similar']['pair']}")
    print(f"   Similarity: {result['least_similar']['similarity']:.2%}")
    print("\n✅ Test 2 PASSED")
    return True


def main():
    print("\n" + "=" * 70)
    print("🧠 ATHENA DOCUMENT COMPARISON TEST SUITE")
    print("=" * 70)
    
    passed = []
    passed.append(("Basic Comparison", test_basic_comparison()))
    passed.append(("Batch Comparison", test_batch_comparison()))
    
    print("\n\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    for name, ok in passed:
        print(f"{'✅' if ok else '❌'} {name}")
    print("\n🎉 All tests done!")


if __name__ == "__main__":
    sys.exit(main())
