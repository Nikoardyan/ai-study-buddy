"""
AI Study Buddy - CLI version
Update TASK 3: Conversation memory.
"""
from src.loader import load_all_from_folder
from src.indexer import (
    index_document,
    get_collection_stats,
    detect_subject
)
from src.generator import generate_answer_streaming
from src.memory import ChatHistory          # ← IMPORT BARU
from src.config import DOCS_FOLDER


def index_all_materials():
    """Index semua materi di folder ./docs."""
    print("=" * 60)
    print("INDEXING MATERI BELAJAR")
    print("=" * 60)

    documents = load_all_from_folder(DOCS_FOLDER)

    if not documents:
        print(f"\nTidak ada dokumen di folder {DOCS_FOLDER}")
        print("Taro PDF/DOCX materi belajar di folder docs/")
        return

    print(f"\nMulai indexing {len(documents)} dokumen...\n")

    for doc in documents:
        subject = detect_subject(doc["filename"])
        print(f"   {doc['filename']}")
        print(f"   Subject: {subject}")
        index_document(
            doc,
            subject=subject,
            grade_level="umum",
            difficulty="intermediate"
        )
        print()

    stats = get_collection_stats()
    print(f"Done! Total: {stats['total_chunks']} chunks ter-index\n")


def chat_streaming():
    """Interactive chat dengan conversation memory."""
    print("=" * 60)
    print("AI STUDY BUDDY - Interactive Chat")
    print("=" * 60)
    print("Tips:")
    print("- Tanya apapun tentang materi belajar lo")
    print("- Sebut subject buat filter: 'soal matematika...'")
    print("- Ketik 'history' untuk lihat history chat")
    print("- Ketik 'clear' untuk hapus history")
    print("- Ketik 'exit' untuk berhenti\n")

    # ← BUAT CHAT HISTORY OBJECT
    history = ChatHistory(max_turns=10)

    while True:
        question = input("\nTanya: ").strip()

        # Commands
        if question.lower() in ["exit", "quit", "keluar"]:
            print("\nSelamat belajar! Semangat ya!")
            break

        if question.lower() == "history":
            history.show()
            continue

        if question.lower() == "clear":
            history.clear()
            print("History chat udah di-reset!")
            continue

        if not question:
            continue

        # Generate & stream jawaban
        print("\nAI Tutor: ", end="", flush=True)
        for word in generate_answer_streaming(
            question=question,
            chat_history=history    # ← KIRIM HISTORY
        ):
            print(word, end="", flush=True)
        print()

        # Info turn count
        print(f"\n[Turn ke-{history.turn_count} | "
              f"History: {len(history.messages)} messages]")


if __name__ == "__main__":
    stats = get_collection_stats()

    if stats["total_chunks"] == 0:
        print("Vector DB kosong, mulai indexing...\n")
        index_all_materials()
    else:
        print(f"Vector DB udah punya {stats['total_chunks']} chunks.")
        print("(Hapus folder chroma_db/ kalau mau re-index)\n")

    stats = get_collection_stats()
    if stats["total_chunks"] == 0:
        print("Tidak ada materi.")
        print("Taro PDF/DOCX di folder ./docs/ lalu run lagi.")
    else:
        chat_streaming()