"""
src/hybrid_retriever.py
Module Hybrid Search: Vector (Chroma) + Keyword (BM25) + RRF
Level 2 Part 3 - RAG Production
"""
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from src.indexer import collection


def tokenize(text: str) -> List[str]:
    """
    Tokenize text simple: lowercase + split by whitespace.
    Untuk produksi besar, pakai library tokenizer (nltk/spacy).
    """
    return text.lower().split()


def bm25_search(
    query: str,
    documents: List[str],
    top_k: int = 10
) -> List[Tuple[int, float]]:
    """
    Keyword search dengan algoritma BM25.
    
    Args:
        query: User question
        documents: List of document texts
        top_k: Berapa hasil diambil
    
    Returns:
        List of (index, score) - sorted by score descending
    """
    if not documents:
        return []
    
    # Tokenize semua docs
    tokenized_docs = [tokenize(doc) for doc in documents]
    
    # Setup BM25
    bm25 = BM25Okapi(tokenized_docs)
    
    # Tokenize query
    tokenized_query = tokenize(query)
    
    # Get scores untuk semua docs
    scores = bm25.get_scores(tokenized_query)
    
    # Sort by score, ambil top_k
    indexed_scores = [(i, score) for i, score in enumerate(scores)]
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    
    return indexed_scores[:top_k]


def reciprocal_rank_fusion(
    vector_results: List[int],
    bm25_results: List[int],
    k: int = 60
) -> List[Tuple[int, float]]:
    """
    Gabungkan hasil dari Vector & BM25 pakai algoritma RRF.
    
    RRF Score = sum(1 / (k + rank)) untuk setiap retriever
    
    Args:
        vector_results: List of indices dari vector search (rank order)
        bm25_results: List of indices dari BM25 (rank order)
        k: Konstanta RRF (default 60 - industry standard)
    
    Returns:
        List of (doc_index, rrf_score) sorted by score
    """
    scores = {}
    
    # Score dari vector results
    for rank, doc_idx in enumerate(vector_results):
        score = 1 / (k + rank + 1)  # +1 karena rank dimulai dari 1
        scores[doc_idx] = scores.get(doc_idx, 0) + score
    
    # Score dari BM25 results
    for rank, doc_idx in enumerate(bm25_results):
        score = 1 / (k + rank + 1)
        scores[doc_idx] = scores.get(doc_idx, 0) + score
    
    # Sort by combined score
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_results


def hybrid_retrieve(
    question: str,
    top_k: int = 20,
    where_filter: dict = None,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4
) -> Tuple[List[str], List[dict]]:
    """
    Hybrid retrieval: Vector + BM25 + RRF merge.
    
    Args:
        question: User question
        top_k: Berapa final results
        where_filter: Filter metadata (subject)
        vector_weight: Weight untuk vector results (0-1)
        bm25_weight: Weight untuk BM25 results (0-1)
    
    Returns:
        Tuple (chunks, metadatas) - top_k hasil terbaik
    """
    # ── STEP 1: VECTOR SEARCH (pakai existing Chroma collection) ──
    query_args = {
        "query_texts": [question],
        "n_results": top_k * 2  # Ambil 2x lipat buat hybrid pool
    }
    if where_filter:
        query_args["where"] = where_filter
    
    try:
        vector_results = collection.query(**query_args)
        all_chunks = vector_results["documents"][0] if vector_results["documents"] else []
        all_metadatas = vector_results["metadatas"][0] if vector_results["metadatas"] else []
    except Exception as e:
        print(f"⚠️ Vector search error: {e}")
        return [], []
    
    if not all_chunks:
        return [], []
    
    # Vector rank: index 0, 1, 2, ... (urutan return = ranking)
    vector_ranks = list(range(len(all_chunks)))
    
    # ── STEP 2: BM25 SEARCH (di chunks yang udah di-filter Chroma) ──
    bm25_results = bm25_search(question, all_chunks, top_k=len(all_chunks))
    bm25_ranks = [idx for idx, _ in bm25_results]
    
    # ── STEP 3: MERGE PAKAI RRF ──
    merged = reciprocal_rank_fusion(vector_ranks, bm25_ranks)
    
    # Ambil top_k dari merged result
    top_indices = [idx for idx, _ in merged[:top_k]]
    
    # Build final results
    final_chunks = [all_chunks[i] for i in top_indices]
    final_metadatas = [all_metadatas[i] for i in top_indices]
    
    print(f"\n[🔀 Hybrid Search: Vector + BM25 → {len(final_chunks)} chunks]")
    
    return final_chunks, final_metadatas


def test_hybrid():
    """Quick test hybrid retriever."""
    query = "Apa rumus turunan x kuadrat?"
    
    print(f"\n🔍 Query: {query}\n")
    
    # Test hybrid
    chunks, metas = hybrid_retrieve(
        question=query,
        top_k=5,
        where_filter={"subject": "matematika"}
    )
    
    print(f"📚 Hybrid Results ({len(chunks)} chunks):")
    for i, (chunk, meta) in enumerate(zip(chunks, metas), 1):
        snippet = chunk[:80] + "..." if len(chunk) > 80 else chunk
        print(f"\n  [{i}] {snippet}")
        print(f"      Source: {meta.get('filename', 'unknown')}, hal {meta.get('page_number', 'N/A')}")


if __name__ == "__main__":
    test_hybrid()