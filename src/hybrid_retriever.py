
"""
src/hybrid_retriever.py

Hybrid Search:
- Chroma vector search
- BM25 keyword search atas seluruh korpus yang sesuai filter
- Weighted Reciprocal Rank Fusion (RRF)
- Deduplication untuk chunk identik

Output hybrid_retrieve tetap:
(chunks, metadatas)
"""

import re
from typing import List, Tuple, Optional

from rank_bm25 import BM25Okapi

from src.indexer import collection


# =========================================================
# TOKENIZATION
# =========================================================

def tokenize(text: str) -> List[str]:
    """
    Tokenisasi sederhana untuk teks Indonesia dan istilah teknis.
    Mempertahankan kata dan token seperti HCI, UX, dan angka.
    """
    if not text:
        return []

    return re.findall(
        r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*",
        text.lower(),
    )


# =========================================================
# BM25 SEARCH
# =========================================================

def bm25_search(
    query: str,
    documents: List[str],
    top_k: int = 10,
) -> List[Tuple[int, float]]:
    """
    Keyword search BM25.

    Mengembalikan pasangan (index dokumen, skor),
    diurutkan dari skor tertinggi. Dokumen dengan skor nol
    tidak dimasukkan sebagai hasil keyword yang relevan.
    """
    if not documents or top_k <= 0:
        return []

    tokenized_docs = [tokenize(doc) for doc in documents]
    tokenized_query = tokenize(query)

    if not tokenized_query:
        return []

    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(tokenized_query)

    indexed_scores = [
        (index, float(score))
        for index, score in enumerate(scores)
        if score > 0
    ]

    indexed_scores.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return indexed_scores[:top_k]


# =========================================================
# WEIGHTED RECIPROCAL RANK FUSION
# =========================================================

def reciprocal_rank_fusion(
    vector_results: List[str],
    bm25_results: List[str],
    k: int = 60,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> List[Tuple[str, float]]:
    """
    Gabungkan ranking vector dan BM25 menggunakan weighted RRF.

    Input berupa ID dokumen dalam urutan ranking.
    Output berupa (document_id, fused_score).
    """
    if k <= 0:
        raise ValueError("Parameter k harus lebih besar dari 0.")

    if vector_weight < 0 or bm25_weight < 0:
        raise ValueError("Bobot tidak boleh negatif.")

    if vector_weight == 0 and bm25_weight == 0:
        raise ValueError("Minimal satu bobot harus lebih besar dari 0.")

    scores = {}

    for rank, document_id in enumerate(vector_results, start=1):
        scores[document_id] = (
            scores.get(document_id, 0.0)
            + vector_weight / (k + rank)
        )

    for rank, document_id in enumerate(bm25_results, start=1):
        scores[document_id] = (
            scores.get(document_id, 0.0)
            + bm25_weight / (k + rank)
        )

    return sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )


# =========================================================
# METADATA FILTER
# =========================================================

def _get_corpus(where_filter: Optional[dict] = None):
    """
    Ambil dokumen dan metadata dari korpus Chroma.
    Filter diterapkan pada kedua jalur pencarian.
    """
    if collection is None:
        return [], [], []

    kwargs = {
        "include": ["documents", "metadatas"],
    }

    if where_filter:
        kwargs["where"] = where_filter

    result = collection.get(**kwargs)

    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []

    # Pastikan setiap dokumen mempunyai metadata.
    if len(metadatas) < len(documents):
        metadatas.extend(
            [{} for _ in range(len(documents) - len(metadatas))]
        )

    # Abaikan dokumen kosong dan pertahankan pasangan ID,
    # teks, serta metadata.
    valid = []

    for doc_id, document, metadata in zip(
        ids,
        documents,
        metadatas,
    ):
        if document and str(document).strip():
            valid.append(
                (
                    str(doc_id),
                    str(document),
                    dict(metadata or {}),
                )
            )

    return (
        [item[0] for item in valid],
        [item[1] for item in valid],
        [item[2] for item in valid],
    )


# =========================================================
# HYBRID RETRIEVAL
# =========================================================

def hybrid_retrieve(
    question: str,
    top_k: int = 20,
    where_filter: Optional[dict] = None,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> Tuple[List[str], List[dict]]:
    """
    Hybrid retrieval dengan vector search dan BM25 independen.

    Vector search mengambil kandidat berdasarkan embedding.
    BM25 mencari kata kunci pada seluruh korpus yang memenuhi
    filter metadata. Weighted RRF menggabungkan ranking keduanya.

    Returns:
        Tuple berisi (chunks, metadatas), dengan urutan berpasangan.
    """
    question = (question or "").strip()

    if not question or top_k <= 0:
        return [], []

    if collection is None:
        print("[Hybrid Search] Collection belum tersedia.")
        return [], []

    if vector_weight < 0 or bm25_weight < 0:
        raise ValueError("Bobot retrieval tidak boleh negatif.")

    if vector_weight == 0 and bm25_weight == 0:
        raise ValueError("Minimal satu bobot harus lebih besar dari 0.")

    # -----------------------------------------------------
    # STEP 1: LOAD CORPUS
    # -----------------------------------------------------

    try:
        corpus_ids, corpus_docs, corpus_metas = _get_corpus(
            where_filter=where_filter,
        )
    except Exception as exc:
        print(f"[Hybrid Search: corpus error] {exc}")
        return [], []

    if not corpus_ids:
        print("[Hybrid Search] Tidak ada dokumen yang cocok.")
        return [], []

    id_to_index = {
        doc_id: index
        for index, doc_id in enumerate(corpus_ids)
    }

    # -----------------------------------------------------
    # STEP 2: VECTOR SEARCH
    # -----------------------------------------------------

    vector_ranked_ids = []

    try:
        # Filter yang sama dipakai untuk vector search.
        # Pool dibatasi agar tidak meminta lebih banyak hasil
        # daripada jumlah dokumen yang tersedia.
        pool_size = min(
            len(corpus_ids),
            max(top_k * 3, top_k),
        )

        query_args = {
            "query_texts": [question],
            "n_results": pool_size,
        }

        if where_filter:
            query_args["where"] = where_filter

        vector_result = collection.query(**query_args)

        result_ids = (
            vector_result.get("ids") or [[]]
        )[0]

        for doc_id in result_ids:
            doc_id = str(doc_id)

            if doc_id in id_to_index and doc_id not in vector_ranked_ids:
                vector_ranked_ids.append(doc_id)

    except Exception as exc:
        print(f"[Hybrid Search: vector error] {exc}")

    # -----------------------------------------------------
    # STEP 3: BM25 SEARCH
    # -----------------------------------------------------

    bm25_ranked_ids = []

    try:
        bm25_results = bm25_search(
            query=question,
            documents=corpus_docs,
            top_k=len(corpus_docs),
        )

        bm25_ranked_ids = [
            corpus_ids[index]
            for index, _score in bm25_results
        ]

    except Exception as exc:
        print(f"[Hybrid Search: BM25 error] {exc}")

    # -----------------------------------------------------
    # STEP 4: WEIGHTED RRF
    # -----------------------------------------------------

    merged = reciprocal_rank_fusion(
        vector_results=vector_ranked_ids,
        bm25_results=bm25_ranked_ids,
        k=60,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
    )

    if not merged:
        print("[Hybrid Search] Kedua jalur tidak menghasilkan kandidat.")
        return [], []

    # -----------------------------------------------------
    # STEP 5: EXACT DEDUPLICATION
    # -----------------------------------------------------

    final_chunks = []
    final_metadatas = []
    seen_texts = set()

    for doc_id, _score in merged:
        index = id_to_index.get(doc_id)

        if index is None:
            continue

        document = corpus_docs[index]
        metadata = corpus_metas[index]

        # Hanya buang teks yang identik.
        # Chunk berbeda pada halaman yang sama tetap diperbolehkan.
        normalized_text = " ".join(document.lower().split())

        if normalized_text in seen_texts:
            continue

        seen_texts.add(normalized_text)
        final_chunks.append(document)
        final_metadatas.append(metadata)

        if len(final_chunks) >= top_k:
            break

    print(
        f"\n[Hybrid Search: Vector + BM25 + Weighted RRF "
        f"→ {len(final_chunks)} chunks]"
    )

    return final_chunks, final_metadatas


# =========================================================
# QUICK TEST
# =========================================================

def test_hybrid():
    """Uji retrieval dengan beberapa pertanyaan IMK."""
    tests = [
        "Apa pengertian IMK?",
        "Apa tujuan IMK?",
        "Apa itu usability?",
    ]

    for question in tests:
        print(f"\n{'=' * 60}")
        print(f"Pertanyaan: {question}")

        chunks, metadatas = hybrid_retrieve(
            question=question,
            top_k=5,
        )

        for index, (chunk, metadata) in enumerate(
            zip(chunks, metadatas),
            start=1,
        ):
            print(
                f"\n[{index}] "
                f"Halaman {metadata.get('page_number', 'N/A')} | "
                f"{metadata.get('filename', 'Unknown')}"
            )
            print(chunk[:400])


if __name__ == "__main__":
    test_hybrid()
