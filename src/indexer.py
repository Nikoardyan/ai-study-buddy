
"""
Indexer - simpan dokumen ke ChromaDB.
Update TASK 2: Auto-detect subject dari nama file.
"""

import os
import re
from datetime import datetime

import chromadb

from src.config import CHROMA_PATH, COLLECTION_NAME
from src.chunker import smart_chunk


def detect_subject(filename: str) -> str:
    """
    Deteksi subject berdasarkan nama file.
    Jika tidak cocok dengan keyword, kembalikan 'umum'.
    """
    name = os.path.splitext(os.path.basename(filename))[0].lower()
    name = re.sub(r"[_\-]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    subject_keywords = {
        "IMK": [
            "imk",
            "interaksi manusia komputer",
            "interaksi manusia dan komputer",
            "human computer interaction",
        ],
        "Matematika": [
            "matematika",
            "math",
            "kalkulus",
        ],
        "Pemrograman": [
            "programming",
            "pemrograman",
            "python",
            "java",
        ],
        "Basis Data": [
            "database",
            "basis data",
            "sql",
        ],
        "Kecerdasan Buatan": [
            "artificial intelligence",
            "kecerdasan buatan",
            "machine learning",
            "deep learning",
        ],
    }

    for subject, keywords in subject_keywords.items():
        for keyword in keywords:
            if keyword in name:
                return subject

    return "umum"


# Setup ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)


def index_document(
    document: dict,
    subject: str = "umum",
    grade_level: str = "umum",
    difficulty: str = "intermediate",
    user_id: str = "public",
):
    """
    Index dokumen ke ChromaDB dengan metadata edutech dan kepemilikan user_id.
    """
    if not subject or not subject.strip():
        subject = "umum"
    if not user_id or not user_id.strip():
        user_id = "public"

    all_chunks = []
    all_metadatas = []
    all_ids = []

    for page in document["pages"]:
        page_num = page["page_number"]
        page_text = page["content"]

        chunks = smart_chunk(page_text)

        for chunk_idx, chunk in enumerate(chunks):
            # chunk_id unik mencakup user_id agar dokumen nama sama antar user tidak bentrok
            chunk_id = (
                f"{user_id}_{document['filename']}"
                f"_p{page_num}_c{chunk_idx}"
            )

            metadata = {
                # Universal & Isolation fields
                "user_id": user_id,
                "filename": document["filename"],
                "format": document["format"],
                "page_number": page_num,
                "chunk_index": chunk_idx,
                "total_pages": document["total_pages"],
                "indexed_at": datetime.now().isoformat(),

                # Edutech custom fields
                "subject": subject,
                "grade_level": grade_level,
                "difficulty": difficulty,
            }

            all_chunks.append(chunk)
            all_metadatas.append(metadata)
            all_ids.append(chunk_id)

    if not all_chunks:
        print(
            f"   Warning: Tidak ada konten "
            f"dari {document['filename']}"
        )
        return

    # Upsert mencegah error jika ID chunk sudah ada.
    collection.upsert(
        documents=all_chunks,
        metadatas=all_metadatas,
        ids=all_ids,
    )

    print(
        f"   OK: {len(all_chunks)} chunks "
        f"| subject: {subject}"
    )


def get_collection_stats() -> dict:
    """Mengambil statistik collection ChromaDB."""
    return {
        "total_chunks": collection.count(),
        "collection_name": COLLECTION_NAME,
    }
