
"""
src/reranker.py

Reranking dokumen:
- Cohere multilingual reranker jika API key tersedia.
- Fallback lokal dengan lexical scoring berbobot jika Cohere
  tidak tersedia atau gagal.

Catatan evaluasi (Oktober 2026):
  Reranker lokal saat ini dinonaktifkan di evaluate_rag.py karena
  lexical scoring berbasis kata kunci kurang cocok untuk pertanyaan
  bahasa Indonesia yang pendek. Hasil hybrid (vector + BM25) digunakan
  sebagai baseline sampai akar masalah ditemukan.

  Untuk mengaktifkan kembali, ubah run_config("hybrid_local_rerank")
  di evaluate_rag.py agar memanggil run_rerank(use_cohere=False).
"""

import os
import re
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

try:
    import cohere

    co = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None
except ImportError:
    co = None


STOPWORDS = {
    "apa", "itu", "yang", "dan", "dari", "dengan",
    "bagaimana", "sebutkan", "jelaskan", "adalah",
    "untuk", "pada", "dalam", "sebuah", "atau",
    "kah", "ya", "di", "ke", "dengan", "secara",
    "the", "is", "of", "and", "a", "an",
}

DEFINITION_PATTERNS = (
    r"\badalah\b",
    r"\bmerupakan\b",
    r"\byaitu\b",
    r"\bdiartikan sebagai\b",
    r"\btingkat kemudahan\b",
    r"\bilmu yang mempelajari\b",
)

TOC_PATTERNS = (
    r"\bdaftar isi\b",
    r"\btable of contents\b",
)


def _tokenize(text: str) -> List[str]:
    """Tokenisasi kata dan istilah yang mengandung tanda hubung."""
    return re.findall(
        r"[a-z0-9]+(?:[-'][a-z0-9]+)*",
        (text or "").lower(),
    )


def _content_terms(query: str) -> List[str]:
    """Hilangkan kata umum, tetapi pertahankan istilah inti."""
    return [
        token
        for token in _tokenize(query)
        if token not in STOPWORDS
    ]


def _local_relevance_score(query: str, document: str) -> float:
    """
    Menghitung relevansi menggunakan:
    - Cakupan istilah query.
    - Kecocokan frasa dan istilah inti.
    - Indikator definisi.
    - Posisi istilah di dokumen.
    - Penalti daftar isi.
    """
    query_tokens = _tokenize(query)
    terms = _content_terms(query)
    doc_tokens = _tokenize(document)

    if not terms or not doc_tokens:
        return 0.0

    normalized_doc = " ".join(doc_tokens)
    doc_terms = set(doc_tokens)
    unique_terms = set(terms)

    # Cakupan istilah query.
    matched_terms = unique_terms & doc_terms
    coverage = len(matched_terms) / len(unique_terms)

    score = coverage

    # Bonus jika istilah inti muncul sebagai frasa berurutan.
    normalized_query = " ".join(terms)

    if len(terms) > 1 and normalized_query in normalized_doc:
        score += 0.35

    # Bonus untuk kecocokan istilah inti di awal dokumen.
    early_terms = set(doc_tokens[:100])
    early_matches = unique_terms & early_terms

    score += 0.20 * (
        len(early_matches) / len(unique_terms)
    )

    # Pertanyaan definisi lebih cocok dengan kalimat definisi.
    query_lower = (query or "").lower()
    asks_definition = (
        "pengertian" in query_lower
        or re.search(r"\bapa itu\b", query_lower) is not None
    )

    if asks_definition:
        definition_found = any(
            re.search(pattern, document.lower())
            for pattern in DEFINITION_PATTERNS
        )

        if definition_found and matched_terms:
            score += 0.30

        # Bonus jika istilah inti diikuti penjelasan definisi.
        for term in unique_terms:
            pattern = (
                rf"\b{re.escape(term)}\b"
                rf".{{0,100}}"
                rf"(?:adalah|merupakan|yaitu|ialah)"
            )
            if re.search(pattern, document.lower()):
                score += 0.20
                break

    # Penalti dokumen yang hanya berfungsi sebagai daftar isi.
    if any(
        re.search(pattern, document.lower())
        for pattern in TOC_PATTERNS
    ):
        score -= 0.50

    # Penalti ringan jika istilah hanya muncul jauh di belakang.
    first_position = min(
        (
            normalized_doc.find(term)
            for term in unique_terms
            if term in normalized_doc
        ),
        default=-1,
    )

    if first_position > 300:
        score -= 0.10

    return max(score, 0.0)


def _local_rerank(
    query: str,
    documents: List[Document],
    top_n: int,
) -> List[Document]:
    """Urutkan dokumen berdasarkan skor relevansi lokal."""
    scored_documents = []

    for index, document in enumerate(documents):
        score = _local_relevance_score(
            query,
            document.page_content,
        )

        scored_documents.append((score, -index, document))

    scored_documents.sort(
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )

    results = []

    for score, _, document in scored_documents[:top_n]:
        # Pertahankan metadata sumber dan metadata lain.
        document.metadata = dict(document.metadata or {})
        document.metadata["local_rerank_score"] = round(
            score, 4
        )
        results.append(document)

    return results


def rerank_documents(
    query: str,
    documents: List[Document],
    top_n: int = 5,
    model: str = "rerank-multilingual-v3.0",
) -> List[Document]:
    """Rerank dokumen menggunakan Cohere atau fallback lokal."""
    if not documents or top_n <= 0:
        return []

    top_n = min(top_n, len(documents))

    if co is None:
        print(
            "[Reranker] Cohere tidak dikonfigurasi; "
            "menggunakan lexical fallback lokal."
        )
        return _local_rerank(query, documents, top_n)

    try:
        results = co.rerank(
            query=query,
            documents=[
                document.page_content for document in documents
            ],
            top_n=top_n,
            model=model,
        )

        reranked_docs = []

        for result in results.results:
            document = documents[result.index]
            document.metadata = dict(document.metadata or {})
            document.metadata["rerank_score"] = (
                result.relevance_score
            )
            reranked_docs.append(document)

        return reranked_docs

    except Exception as exc:
        print(
            f"[Reranker] Cohere gagal ({exc}); "
            "menggunakan lexical fallback lokal."
        )
        return _local_rerank(query, documents, top_n)


def test_reranker():
    """Tes dasar urutan dokumen."""
    test_docs = [
        Document(
            page_content=(
                "Daftar Isi: Pengertian IMK, Usability, "
                "dan Struktur Navigasi."
            )
        ),
        Document(
            page_content=(
                "Pengertian dan Tujuan IMK. IMK adalah ilmu "
                "yang mempelajari interaksi manusia dengan "
                "komputer serta merancang sistem efektif, "
                "efisien, dan nyaman digunakan."
            )
        ),
        Document(
            page_content=(
                "Usability adalah tingkat kemudahan sebuah "
                "sistem digunakan untuk mencapai tujuan "
                "secara efektif, efisien, dan memuaskan."
            )
        ),
        Document(
            page_content=(
                "WCAG memiliki empat prinsip POUR: "
                "Perceivable, Operable, Understandable, "
                "dan Robust."
            )
        ),
    ]

    tests = [
        "Apa pengertian IMK?",
        "Apa tujuan IMK?",
        "Apa itu usability?",
    ]

    for query in tests:
        print(f"\nPertanyaan: {query}")

        results = rerank_documents(
            query=query,
            documents=test_docs,
            top_n=4,
        )

        for index, document in enumerate(results, start=1):
            score = document.metadata.get(
                "rerank_score",
                document.metadata.get("local_rerank_score", 0),
            )
            print(
                f"{index}. skor={score:.4f} | "
                f"{document.page_content[:150]}"
            )


if __name__ == "__main__":
    test_reranker()
