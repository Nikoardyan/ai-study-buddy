"""
Smart Router untuk Cost Optimization

Routing query ke model yang tepat berdasarkan complexity:
- SIMPLE   → Llama-8B (cepat, murah ~$0.05/1M tokens)
- MEDIUM   → Llama-70B (balanced ~$0.59/1M tokens)
- COMPLEX  → Llama-70B + extra context

WHY: Bisa hemat 40-60% cost dengan classify query yang
     simple ke model murah, complex ke model mahal.

Author: Niko Ardyan
"""

# ===== CONFIGURATION AS DATA (Anti vibe-coding pattern) =====
# Bisa di-tune tanpa ubah logic code

MODELS = {
    "simple": {
        "name": "llama-3.1-8b-instant",
        "max_tokens": 512,
        "temperature": 0.3,
        "cost_per_1m_tokens": 0.05,
        "description": "Fast & cheap untuk simple queries",
    },
    "medium": {
        "name": "llama-3.3-70b-versatile",
        "max_tokens": 1024,
        "temperature": 0.5,
        "cost_per_1m_tokens": 0.59,
        "description": "Balanced untuk regular queries",
    },
    "complex": {
        "name": "llama-3.3-70b-versatile",
        "max_tokens": 2048,
        "temperature": 0.7,
        "cost_per_1m_tokens": 0.59,
        "description": "Deep reasoning untuk complex queries",
    },
}


# ===== HEURISTIC KEYWORDS =====
# Bisa di-customize sesuai domain (educational, medical, dll)

COMPLEX_INDICATORS = [
    # Bahasa Indonesia
    "bandingkan", "perbandingan",
    "analisis", "analisa",
    "hubungan", "relasi", "korelasi",
    "perbedaan", "persamaan",
    "kenapa", "mengapa", "alasan",
    "jelaskan secara detail",
    "step by step",
    "langkah demi langkah",
    "berikan contoh",
    "uraikan",
    
    # English
    "compare", "comparison",
    "analyze", "analysis",
    "relationship", "correlation",
    "difference", "similarity",
    "why", "reason",
    "explain in detail",
    "give example",
    "describe in depth",
]

SIMPLE_INDICATORS = [
    # Bahasa Indonesia
    "apa itu", "apa arti",
    "definisi", "pengertian",
    "kapan", "tahun berapa",
    "siapa", "siapakah",
    "dimana", "di mana",
    "sebutkan",
    
    # English
    "what is", "what are",
    "definition of",
    "when", "what year",
    "who", "who is",
    "where",
    "list",
]


# ===== CORE FUNCTIONS =====

def classify_query(query: str) -> str:
    """
    Classify query into: simple / medium / complex.
    
    WHAT: Determine query complexity tier
    WHY: Untuk routing ke model yang tepat
         (Llama-8B untuk simple = hemat 90%+ cost vs 70B)
    
    Logic (priority order):
    1. Pendek (≤8 kata) + simple indicator → SIMPLE
    2. Ada complex indicator → COMPLEX
    3. Panjang (>20 kata) → COMPLEX
    4. Default → MEDIUM
    
    Args:
        query: User query string
        
    Returns:
        Complexity tier: "simple", "medium", or "complex"
    """
    if not query or not query.strip():
        return "medium"  # Default safe
    
    query_lower = query.lower().strip()
    query_length = len(query.split())
    
    # Rule 1: Pendek + simple indicator → SIMPLE
    if query_length <= 8:
        for indicator in SIMPLE_INDICATORS:
            if indicator in query_lower:
                return "simple"
    
    # Rule 2: Ada complex indicator → COMPLEX
    for indicator in COMPLEX_INDICATORS:
        if indicator in query_lower:
            return "complex"
    
    # Rule 3: Query panjang → COMPLEX
    if query_length > 20:
        return "complex"
    
    # Rule 4: Default → MEDIUM
    return "medium"


def route_query(query: str) -> dict:
    """
    Route query to appropriate model configuration.
    
    WHAT: Combine classify + lookup config
    WHY: Single entry point untuk integrasi ke generator.py
    
    Args:
        query: User query string
        
    Returns:
        Dict with full model config:
        - name: model identifier
        - max_tokens: int
        - temperature: float
        - cost_per_1m_tokens: float
        - tier: "simple" / "medium" / "complex"
        - description: str
    """
    tier = classify_query(query)
    config = MODELS[tier].copy()
    config["tier"] = tier
    return config


def estimate_cost(query: str, expected_response_tokens: int = 500) -> dict:
    """
    Estimate cost untuk a query.
    
    WHAT: Calculate expected cost based on tier
    WHY: Cost transparency untuk monitoring
    
    Args:
        query: User query
        expected_response_tokens: Estimated response length
        
    Returns:
        Dict with cost breakdown
    """
    config = route_query(query)
    # Rough estimate: 1 word ≈ 1.3 tokens
    query_tokens = int(len(query.split()) * 1.3)
    total_tokens = query_tokens + expected_response_tokens
    
    cost_usd = (total_tokens / 1_000_000) * config["cost_per_1m_tokens"]
    
    return {
        "tier": config["tier"],
        "model": config["name"],
        "query_tokens": query_tokens,
        "response_tokens": expected_response_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "cost_idr": cost_usd * 16500,  # Approx IDR rate
    }


# ===== STANDALONE TEST =====
if __name__ == "__main__":
    test_queries = [
        # SIMPLE — pendek, definition queries
        "Apa itu mitosis?",
        "Definisi enzim",
        "Kapan Indonesia merdeka?",
        "Siapa penemu DNA?",
        
        # MEDIUM — regular question
        "Jelaskan proses fotosintesis",
        "Bagaimana cara kerja sistem pencernaan",
        "Tahapan replikasi DNA",
        
        # COMPLEX — comparison, analysis, deep explain
        "Bandingkan respirasi aerob dan anaerob",
        "Jelaskan hubungan antara DNA dan protein synthesis dengan contoh detail",
        "Analisis perbedaan mitosis dan meiosis berdasarkan tahapan dan tujuan",
        "Mengapa enzim hanya bekerja pada substrat tertentu? Jelaskan dengan teori lock and key",
    ]
    
    print("=" * 70)
    print("                  🤖 SMART ROUTER TEST")
    print("=" * 70)
    print()
    
    tier_emoji = {"simple": "🟢", "medium": "🟡", "complex": "🔴"}
    tier_count = {"simple": 0, "medium": 0, "complex": 0}
    total_cost = 0.0
    
    for i, query in enumerate(test_queries, 1):
        cost_info = estimate_cost(query)
        emoji = tier_emoji[cost_info["tier"]]
        tier_count[cost_info["tier"]] += 1
        total_cost += cost_info["cost_usd"]
        
        print(f"{i}. {emoji} Tier: {cost_info['tier'].upper()}")
        print(f"   Query: {query}")
        print(f"   Model: {cost_info['model']}")
        print(f"   Tokens: ~{cost_info['total_tokens']} | Cost: ${cost_info['cost_usd']:.6f}")
        print()
    
    print("=" * 70)
    print("                       📊 SUMMARY")
    print("=" * 70)
    print(f"Total queries:   {len(test_queries)}")
    print(f"🟢 Simple:        {tier_count['simple']} queries")
    print(f"🟡 Medium:        {tier_count['medium']} queries")
    print(f"🔴 Complex:       {tier_count['complex']} queries")
    print(f"💰 Total cost:    ${total_cost:.6f} (~Rp {total_cost * 16500:.2f})")
    print()
    print("💡 Cost saving estimasi:")
    print("   - Tanpa router (semua pakai 70B): higher cost")
    print("   - Dengan router (simple → 8B):     ~50% lebih murah!")
    print("=" * 70)