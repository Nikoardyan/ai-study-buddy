"""
Retrieval Evaluator - AI Study Buddy
Menguji hasil Vector Search dan Hybrid Search.
Tidak mengubah UI atau database.
"""

from src.indexer import collection
from src.hybrid_retriever import hybrid_retrieve


def vector_search(question, top_k=5):
    """Mengambil top-k hasil Vector Search."""

    result = collection.query(
        query_texts=[question],
        n_results=min(top_k, collection.count()),
    )

    docs = result["documents"][0] if result["documents"] else []
    metas = result["metadatas"][0] if result["metadatas"] else []

    return docs, metas


def show_results(label, documents, metadatas):
    print(f"\n--- {label} ---")

    if not documents:
        print("Tidak ada dokumen ditemukan.")
        return

    for rank, (doc, meta) in enumerate(
        zip(documents, metadatas), start=1
    ):
        print(
            f"\nRank {rank} | "
            f"File: {meta.get('filename', 'unknown')} | "
            f"Halaman: {meta.get('page_number', 'N/A')}"
        )
        print(doc[:250].replace("\n", " "))


def main():
    total = collection.count()

    print("\n=== AI STUDY BUDDY: RETRIEVAL TEST ===")
    print(f"Total chunks di ChromaDB: {total}")

    if total == 0:
        print("Database kosong. Upload dan index dokumen melalui UI.")
        return

    question = input(
        "\nMasukkan pertanyaan berdasarkan dokumen lu: "
    ).strip()

    if not question:
        print("Pertanyaan tidak boleh kosong.")
        return

    top_k = 5

    # 1. Vector search
    vector_docs, vector_metas = vector_search(
        question, top_k=top_k
    )

    # 2. Hybrid search
    hybrid_docs, hybrid_metas = hybrid_retrieve(
        question, top_k=top_k
    )

    # 3. Tampilkan hasil untuk inspeksi manual
    show_results("VECTOR SEARCH", vector_docs, vector_metas)
    show_results("HYBRID SEARCH", hybrid_docs, hybrid_metas)

    print("\n=== RINGKASAN ===")
    print(f"Vector Search: {len(vector_docs)} chunks")
    print(f"Hybrid Search: {len(hybrid_docs)} chunks")
    print(
        "\nPeriksa apakah hasil teratas benar-benar menjawab "
        "pertanyaan. Ini tes inspeksi hasil, belum skor akurasi."
    )


if __name__ == "__main__":
    main()