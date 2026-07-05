"""
src/cache_manager.py
Caching Strategy untuk RAG Production
Level 2 Part 3 - Hemat cost 80%

Implementations:
- Level 1: Exact Match Cache (key = query string)
- Level 2: Semantic Cache (key = embedding similarity)
"""
import hashlib
import json
import os
from typing import Optional
from pathlib import Path
import diskcache as dc
from sentence_transformers import SentenceTransformer
import numpy as np


# ── SETUP ─────────────────────────────────────────────────────
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

# Cache untuk exact match (Level 1)
exact_cache = dc.Cache(str(CACHE_DIR / "exact_cache"))

# Cache untuk semantic (Level 2) - simpan (embedding, response) pairs
semantic_cache = dc.Cache(str(CACHE_DIR / "semantic_cache"))

# Embedding model untuk semantic similarity
# Pakai model yang cepat & multilingual
_embedder = None

def get_embedder():
    """Lazy load embedder (load 1x aja)."""
    global _embedder
    if _embedder is None:
        print("[Cache] Loading embedding model...")
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedder


# ── HELPER FUNCTIONS ──────────────────────────────────────────
def _hash_key(text: str, context_key: str = "") -> str:
    """Hash text jadi key cache yang konsisten."""
    return hashlib.md5((text.lower().strip() + context_key).encode()).hexdigest()


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity antara 2 vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))


# ── LEVEL 1: EXACT MATCH CACHE ────────────────────────────────
def get_exact_cache(query: str, context_key: str = "") -> Optional[str]:
    """
    Cek apakah query persis ada di cache.
    
    Returns:
        Cached response kalau ada, None kalau ga ada
    """
    key = _hash_key(query, context_key)
    return exact_cache.get(key)


def set_exact_cache(query: str, response: str, context_key: str = ""):
    """Simpan response ke exact cache."""
    key = _hash_key(query, context_key)
    exact_cache.set(key, response, expire=86400)  # Expire 24 jam


# ── LEVEL 2: SEMANTIC CACHE ───────────────────────────────────
def get_semantic_cache(
    query: str,
    threshold: float = 0.85,    # ← Diturunin dari 0.92
    context_key: str = ""
) -> Optional[str]:
    """
    Cek apakah ada query MIRIP di cache.
    
    Args:
        query: User question
        threshold: Minimum similarity (0-1). Default 0.85 = 85% mirip
    
    Returns:
        Cached response kalau mirip, None kalau ga ada match
    """
    # Compute embedding query
    embedder = get_embedder()
    query_embedding = embedder.encode([query])[0]
    
    # Cari di semua cached entries
    best_similarity = 0.0
    best_response = None
    best_query = None
    
    for key in semantic_cache.iterkeys():
        try:
            entry = semantic_cache.get(key)
            if not entry:
                continue
            if entry.get("context_key", "") != context_key:
                continue
            
            cached_embedding = np.array(entry["embedding"])
            similarity = _cosine_similarity(query_embedding, cached_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_response = entry["response"]
                best_query = entry["query"]
        except Exception as e:
            continue
    
    # Debug: print similarity walaupun MISS (biar tau gap-nya)
    if best_query:
        if best_similarity >= threshold:
            print(f"[💾 Semantic Cache HIT] similarity={best_similarity:.3f}")
            print(f"   Original query: '{best_query}'")
            return best_response
        else:
            # Bukan HIT, tapi tampilin info closest match buat debugging
            print(f"[Cache] Closest match: '{best_query}' (similarity={best_similarity:.3f}, threshold={threshold})")
    
    return None


def set_semantic_cache(query: str, response: str, context_key: str = ""):
    """Simpan ke semantic cache dengan embedding."""
    embedder = get_embedder()
    embedding = embedder.encode([query])[0]
    
    key = _hash_key(query, context_key)
    semantic_cache.set(key, {
        "query": query,
        "context_key": context_key,
        "embedding": embedding.tolist(),  # Convert numpy → list (serializable)
        "response": response
    }, expire=86400)  # Expire 24 jam


# ── UNIFIED INTERFACE ─────────────────────────────────────────
def check_cache(query: str, context_key: str = "") -> Optional[str]:
    """
    Cek cache dengan strategy: exact match dulu, baru semantic.
    
    Returns:
        Cached response kalau hit, None kalau miss
    """
    # Level 1: Exact match (fastest)
    cached = get_exact_cache(query, context_key)
    if cached:
        print(f"[💾 Exact Cache HIT]")
        return cached
    
    # Level 2: Semantic similarity (smarter)
    cached = get_semantic_cache(query, threshold=0.85, context_key=context_key)
    if cached:
        return cached
    
    return None


def save_to_cache(query: str, response: str, context_key: str = ""):
    """Simpan ke both exact dan semantic cache."""
    set_exact_cache(query, response, context_key)
    set_semantic_cache(query, response, context_key)


def get_cache_stats() -> dict:
    """Stats untuk monitoring cache."""
    return {
        "exact_cache_size": len(exact_cache),
        "semantic_cache_size": len(semantic_cache),
        "cache_dir": str(CACHE_DIR),
    }


def clear_cache():
    """Clear semua cache. Pakai pas testing."""
    exact_cache.clear()
    semantic_cache.clear()
    print("[Cache] Cleared all caches")


# ── TEST ──────────────────────────────────────────────────────
def test_cache():
    """Test caching system."""
    print("\n=== Testing Cache System ===\n")
    
    # Clear dulu
    clear_cache()
    
    # Test 1: First query (MISS)
    query1 = "Apa itu turunan dalam matematika?"
    response1 = "Turunan adalah laju perubahan suatu fungsi..."
    
    result = check_cache(query1)
    print(f"Test 1: '{query1}'")
    print(f"  Cache: {'HIT' if result else 'MISS'} (expected MISS)\n")
    
    # Simpan
    save_to_cache(query1, response1)
    print(f"  → Saved to cache\n")
    
    # Test 2: Exact same query (EXACT HIT)
    print(f"Test 2: '{query1}' (sama persis)")
    result = check_cache(query1)
    print(f"  Cache: {'HIT' if result else 'MISS'} (expected EXACT HIT)\n")
    
    # Test 3: Similar query (SEMANTIC HIT)
    query2 = "Jelaskan apa itu turunan matematika dong"
    print(f"Test 3: '{query2}' (mirip)")
    result = check_cache(query2)
    print(f"  Cache: {'HIT' if result else 'MISS'} (expected SEMANTIC HIT)\n")
    
    # Test 4: Different query (MISS)
    query3 = "Apa itu Python programming?"
    print(f"Test 4: '{query3}' (beda total)")
    result = check_cache(query3)
    print(f"  Cache: {'HIT' if result else 'MISS'} (expected MISS)\n")
    
    # Stats
    stats = get_cache_stats()
    print(f"\n=== Cache Stats ===")
    print(f"Exact cache: {stats['exact_cache_size']} entries")
    print(f"Semantic cache: {stats['semantic_cache_size']} entries")


if __name__ == "__main__":
    test_cache()