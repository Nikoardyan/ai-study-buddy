"""
Indexer - simpan dokumen ke ChromaDB.
Update TASK 2: Auto-detect subject dari nama file.
"""
from datetime import datetime
import chromadb
from src.config import CHROMA_PATH, COLLECTION_NAME
from src.chunker import smart_chunk


# Setup ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def detect_subject(filename: str) -> str:
    """
    Auto-detect subject dari nama file.

    Contoh:
    '01_matematika_kalkulus.pdf'  -> 'matematika'
    '02_fisika_hukum_newton.pdf'  -> 'fisika'
    '03_pengenalan_python.pdf'    -> 'python'
    """
    filename_lower = filename.lower()

    subject_keywords = {
        "matematika": ["matematika", "math", "kalkulus",
                       "aljabar", "geometri", "statistik"],
        "fisika":     ["fisika", "physics", "newton",
                       "mekanika", "termodinamika", "gelombang"],
        "kimia":      ["kimia", "chemistry", "reaksi", "molekul"],
        "biologi":    ["biologi", "biology", "sel", "genetika"],
        "python":     ["python", "programming", "coding",
                       "pemrograman", "algoritma"],
        "sejarah":    ["sejarah", "history"],
        "ekonomi":    ["ekonomi", "economics"],
        "bahasa":     ["bahasa", "grammar", "sastra"],
    }

    for subject, keywords in subject_keywords.items():
        for keyword in keywords:
            if keyword in filename_lower:
                return subject

    return "general"


def index_document(
    document: dict,
    subject: str = None,
    grade_level: str = "umum",
    difficulty: str = "intermediate"
):
    """
    Index dokumen ke ChromaDB dengan metadata edutech.
    Kalau subject tidak dikasih, auto-detect dari nama file.
    """
    # Auto-detect kalau subject tidak dikasih
    if not subject:
        subject = detect_subject(document["filename"])

    all_chunks = []
    all_metadatas = []
    all_ids = []

    for page in document["pages"]:
        page_num = page["page_number"]
        page_text = page["content"]

        chunks = smart_chunk(page_text)

        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"{document['filename']}_p{page_num}_c{chunk_idx}"

            metadata = {
                # Universal fields
                "filename":    document["filename"],
                "format":      document["format"],
                "page_number": page_num,
                "chunk_index": chunk_idx,
                "total_pages": document["total_pages"],
                "indexed_at":  datetime.now().isoformat(),

                # Edutech custom fields
                "subject":     subject,
                "grade_level": grade_level,
                "difficulty":  difficulty,
            }

            all_chunks.append(chunk)
            all_metadatas.append(metadata)
            all_ids.append(chunk_id)

    if all_chunks:
        collection.add(
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids
        )
        print(f"   OK: {len(all_chunks)} chunks | subject: {subject}")
    else:
        print(f"   Warning: Tidak ada konten dari {document['filename']}")


def get_collection_stats() -> dict:
    """Get statistik collection."""
    return {
        "total_chunks": collection.count(),
        "collection_name": COLLECTION_NAME
    }