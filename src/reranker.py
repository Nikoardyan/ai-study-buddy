"""
src/reranker.py
Module untuk Reranking - boost akurasi retrieval
"""
import os
from dotenv import load_dotenv
import cohere
from typing import List
from langchain_core.documents import Document

# Load .env file - WAJIB!
load_dotenv()

# Initialize Cohere client
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY not set in .env file!")

# Pakai Client (BUKAN ClientV2)
co = cohere.Client(COHERE_API_KEY)


def rerank_documents(
    query: str,
    documents: List[Document],
    top_n: int = 5,
    model: str = "rerank-multilingual-v3.0"
) -> List[Document]:
    """
    Rerank documents berdasarkan relevansi ke query.
    """
    if not documents:
        return []
    
    docs_text = [doc.page_content for doc in documents]
    
    try:
        results = co.rerank(
            query=query,
            documents=docs_text,
            top_n=min(top_n, len(documents)),
            model=model
        )
        
        reranked_docs = []
        for result in results.results:
            original_doc = documents[result.index]
            original_doc.metadata["rerank_score"] = result.relevance_score
            reranked_docs.append(original_doc)
        
        return reranked_docs
    
    except Exception as e:
        print(f"⚠️ Reranking failed: {e}")
        return documents[:top_n]


def test_reranker():
    """Quick test untuk reranker."""
    test_docs = [
        Document(page_content="Turunan dari x kuadrat adalah 2x."),
        Document(page_content="Aljabar adalah cabang matematika."),
        Document(page_content="Rumus turunan: d/dx(x^n) = n*x^(n-1)."),
        Document(page_content="Trigonometri membahas sin, cos, tan."),
    ]
    
    query = "Apa rumus turunan x kuadrat?"
    
    print(f"\n🔍 Query: {query}")
    print(f"📚 Original order ({len(test_docs)} docs):")
    for i, doc in enumerate(test_docs):
        print(f"  {i+1}. {doc.page_content}")
    
    reranked = rerank_documents(query, test_docs, top_n=3)
    
    print(f"\n✨ Reranked order (top 3):")
    for i, doc in enumerate(reranked):
        score = doc.metadata.get("rerank_score", 0)
        print(f"  {i+1}. [{score:.3f}] {doc.page_content}")


if __name__ == "__main__":
    test_reranker()