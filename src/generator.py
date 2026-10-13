
import hashlib
import re
from typing import Optional

from openai import OpenAI
from langchain_core.documents import Document

from src.config import (
    API_KEY,
    BASE_URL,
    LLM_MODEL,
    TEMPERATURE,
    MAX_TOKENS,
    SYSTEM_PROMPT,
)
from src.indexer import collection
from src.memory import ChatHistory
from src.hybrid_retriever import hybrid_retrieve
from src.reranker import rerank_documents
from src.cache_manager import check_cache, save_to_cache
import time


client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# =========================================================
# CITATION HELPERS
# =========================================================

def _extract_valid_citations(
    answer: str,
    total_chunks: int,
) -> list[int]:
    """Ambil nomor sitasi yang benar-benar tersedia di konteks.
    Mendukung format ``[1]`` maupun ``【1】`` serta variasi spasi.
    """
    if not answer or total_chunks <= 0:
        return []

    # Cari semua pola sitasi yang mungkin muncul.
    patterns = [
        r"\[\s*(?:【\s*)?(\d+)(?:\s*】)?\s*\]",
        r"【\s*(\d+)\s*】",
    ]
    matches = []
    for pat in patterns:
        matches.extend(re.findall(pat, answer))

    # Filter nomor yang berada dalam rentang chunk yang ada.
    valid = {
        int(num)
        for num in matches
        if 1 <= int(num) <= total_chunks
    }
    return sorted(valid)


def _get_filename(metadata: dict) -> Optional[str]:
    for key in ("filename", "source", "file_name"):
        value = (metadata or {}).get(key)
        if value:
            return str(value)
    return None


def _get_page(metadata: dict):
    for key in ("page_number", "page"):
        value = (metadata or {}).get(key)
        if value is not None:
            return value
    return None


def _format_citations(
    citation_numbers: list[int],
    metadatas: list[dict],
    title: str = "**Sumber materi:**",
) -> str:
    """Format sitasi berdasarkan metadata asli dokumen."""
    sources = {}

    for number in citation_numbers:
        index = number - 1

        if index < 0 or index >= len(metadatas):
            continue

        metadata = metadatas[index] or {}
        filename = _get_filename(metadata)
        page = _get_page(metadata)

        if not filename or page is None:
            continue

        key = (filename, str(page))
        sources.setdefault(key, [])

        if number not in sources[key]:
            sources[key].append(number)

    if not sources:
        return ""

    lines = []

    for (filename, page), numbers in sources.items():
        refs = ", ".join(f"[{n}]" for n in numbers)
        lines.append(
            f"{refs} {filename}, halaman {page}"
        )

    return title + "\n" + "\n".join(lines)


def _split_cached_sources(
    cached_response: str,
) -> tuple[str, str]:
    """Pisahkan jawaban dan daftar sumber dari cache."""
    markers = (
        "**Sumber materi:**",
        "**Sumber hasil pencarian (sitasi AI tidak tersedia):**",
    )

    for marker in markers:
        if marker in cached_response:
            answer, sources = cached_response.split(marker, 1)
            return answer.rstrip(), marker + sources.rstrip()

    return cached_response.strip(), ""


def _make_context_key(
    where_filter: Optional[dict],
    documents: list[str],
    metadatas: list[dict],
) -> str:
    """Buat cache key berdasarkan versi dan konteks."""
    parts = [
        "citation-v4-no-fallback-sources",
        str(where_filter or {}),
    ]

    for document, metadata in zip(documents, metadatas):
        metadata = metadata or {}
        parts.append(
            f"{document}|"
            f"{_get_filename(metadata)}|"
            f"{_get_page(metadata)}"
        )

    return hashlib.sha256(
        "\n".join(parts).encode("utf-8")
    ).hexdigest()


def _answer_says_not_found(answer: str) -> bool:
    """Kenali jawaban yang menyatakan informasi tidak ditemukan."""
    text = (answer or "").lower()

    indicators = (
        "tidak ada informasi mengenai",
        "tidak ada informasi tentang",
        "tidak ditemukan dalam materi",
        "tidak ditemukan di dalam materi",
        "tidak terdapat dalam materi",
        "tidak terdapat di dalam materi",
        "tidak tersedia dalam materi",
        "tidak tersedia di dalam materi",
        "tidak membahas",
        "tidak dibahas dalam materi",
        "tidak tersedia dalam pdf",
        "tidak tersedia di pdf",
        "materi yang tersedia dalam pdf hanya membahas",
        "saya tidak menemukan",
        "tidak dapat menjawab berdasarkan materi",
    )

    return any(phrase in text for phrase in indicators)


# =========================================================
# QUERY REWRITING
# =========================================================

def rewrite_query(
    question: str,
    chat_history: Optional[ChatHistory] = None,
) -> str:
    """Perjelas pertanyaan lanjutan berdasarkan riwayat chat."""
    if (
        chat_history is None
        or not getattr(chat_history, "messages", None)
    ):
        return question

    try:
        history = chat_history.get_messages(SYSTEM_PROMPT)

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ubah pertanyaan lanjutan menjadi pertanyaan "
                        "mandiri berdasarkan riwayat percakapan. "
                        "Pertahankan maksud pengguna. Keluarkan hanya "
                        "pertanyaan hasil revisi. Jika pertanyaan sudah "
                        "mandiri, kembalikan tanpa perubahan. Jangan memotong "
                        "kalimat atau menambahkan penjelasan lain."
                    ),
                },
                *history[-6:],
                {
                    "role": "user",
                    "content": question,
                },
            ],
            temperature=0,
            max_tokens=300,
        )

        choice = response.choices[0]
        # Jika respon terpotong karena batas token (finish_reason == 'length'), fallback ke pertanyaan asli
        if choice.finish_reason == "length":
            print(f"[Query rewrite warning] Truncated by token limit; fallback to original.")
            return question

        rewritten = (choice.message.content or "").strip()

        # Validasi: jika hasil rewrite terlalu pendek dibanding aslinya (misal < 5 karakter padahal aslinya panjang), gunakan original
        if rewritten and len(rewritten) >= min(len(question) * 0.5, 10):
            return rewritten

    except Exception as exc:
        print(f"[Query rewrite fallback] {exc}")

    return question


# =========================================================
# RETRIEVAL + RERANKING
# =========================================================

def retrieve_with_rerank(
    question: str,
    initial_k: int = 20,
    final_n: int = 5,
    where_filter: Optional[dict] = None,
) -> tuple[list[str], list[dict]]:
    """Cari dokumen dan lakukan reranking dengan metadata berpasangan."""
    try:
        documents, metadatas = hybrid_retrieve(
            question,
            top_k=initial_k,
            where_filter=where_filter,
        )

        if not documents:
            return [], []

        documents = list(documents)
        metadatas = list(metadatas or [])

        if len(metadatas) < len(documents):
            metadatas.extend(
                [{} for _ in range(len(documents) - len(metadatas))]
            )

        langchain_documents = [
            Document(
                page_content=str(document),
                metadata=metadata or {},
            )
            for document, metadata in zip(
                documents, metadatas
            )
        ]

        try:
            reranked = rerank_documents(
                query=question,
                documents=langchain_documents,
                top_n=final_n,
            )

            if reranked:
                langchain_documents = reranked

        except Exception as exc:
            print(f"[Rerank fallback] {exc}")

        selected = langchain_documents[:final_n]

        return (
            [doc.page_content for doc in selected],
            [dict(doc.metadata or {}) for doc in selected],
        )

    except Exception as exc:
        print(f"[Hybrid retrieval error] {exc}")
        return [], []


# =========================================================
# MAIN GENERATOR
# =========================================================

def generate_answer_streaming(
    question: str,
    chat_history: Optional[ChatHistory] = None,
    subject_filter: Optional[str] = None,
    filename_filter: Optional[str] = None,
    user_id: Optional[str] = "public",
):
    """Hasilkan jawaban streaming dengan sitasi metadata PDF dan isolasi dokumen user_id."""
    question = (question or "").strip()
    user_id = (user_id or "public").strip()

    if not question:
        yield "Silakan masukkan pertanyaan terlebih dahulu."
        return

    if collection is None:
        yield "Database materi belum tersedia."
        return

    try:
        if collection.count() == 0:
            yield (
                "Belum ada materi yang diindeks. "
                "Silakan indeks PDF terlebih dahulu."
            )
            return

    except Exception as exc:
        print(f"[Collection error] {exc}")
        yield "Gagal memeriksa database materi."
        return

    # Metadata filter: isolasi dokumen berdasarkan kepemilikan user_id atau dokumen materi publik
    if user_id == "public":
        user_clause = {"user_id": "public"}
    else:
        user_clause = {
            "$or": [
                {"user_id": user_id},
                {"user_id": "public"},
            ]
        }

    filter_conditions = [user_clause]

    if subject_filter:
        filter_conditions.append({"subject": subject_filter})
    if filename_filter:
        filter_conditions.append({"filename": filename_filter})

    if len(filter_conditions) == 1:
        where_filter = filter_conditions[0]
    else:
        where_filter = {"$and": filter_conditions}

    # Rewrite pertanyaan berdasarkan riwayat sebelumnya.
    search_question = rewrite_query(question, chat_history)
    start_retrieval = time.time()

    documents, metadatas = retrieve_with_rerank(
        question=search_question,
        initial_k=20,
        final_n=5,
        where_filter=where_filter,
    )
    retrieval_ms = (time.time() - start_retrieval) * 1000

    # Jangan fallback ke None tanpa filter user_clause untuk mencegah kebocoran dokumen!
    if not documents and (subject_filter or filename_filter):
        print(
            "[Retrieval fallback] Tidak ada hasil dengan filter subject/filename; "
            "mencoba dengan filter user_clause saja."
        )

        documents, metadatas = retrieve_with_rerank(
            question=search_question,
            initial_k=20,
            final_n=5,
            where_filter=user_clause,
        )

    if not documents:
        message = "Maaf, materi yang relevan belum ditemukan."

        if chat_history:
            chat_history.add_user(question)
            chat_history.add_assistant(message)

        yield message
        return

    context_key = _make_context_key(
        where_filter,
        documents,
        metadatas,
    )

    # Cache
    try:
        cached_response = check_cache(
            question,
            context_key=context_key,
        )
    except Exception as exc:
        print(f"[Cache read error] {exc}")
        cached_response = None

    if cached_response:
        print("[Cache HIT: context matched]")

        cached_answer, cached_sources = _split_cached_sources(
            str(cached_response)
        )

        if chat_history:
            chat_history.add_user(question)
            chat_history.add_assistant(cached_answer)

        yield cached_answer

        if cached_sources:
            yield "\n\n" + cached_sources

        return

    # Build context
    context_parts = []

    for index, (document, metadata) in enumerate(
        zip(documents, metadatas),
        start=1,
    ):
        metadata = metadata or {}

        filename = _get_filename(metadata) or "Tidak diketahui"
        page = _get_page(metadata)
        page_label = (
            str(page) if page is not None else "Tidak diketahui"
        )

        context_parts.append(
            f"[{index}] File: {filename}, "
            f"halaman: {page_label}\n{document}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
Jawab pertanyaan pengguna berdasarkan konteks materi yang diberikan.

ATURAN WAJIB:
1. Gunakan konteks sebagai sumber utama.
2. Jangan mengarang fakta atau sitasi.
3. Berikan sitasi [1], [2], dan seterusnya hanya pada klaim
   yang benar-benar didukung oleh chunk bernomor tersebut.
4. Nomor sitasi merujuk pada nomor chunk di KONTEKS,
   bukan nomor halaman PDF.
5. Jika konteks tidak memuat jawaban, katakan dengan jelas
   bahwa informasi tidak ditemukan dalam PDF.
6. Jika informasi tidak ditemukan, jangan menambahkan sitasi.
7. Jawab dalam Bahasa Indonesia yang jelas.
8. Jangan membuat daftar sumber sendiri. Sistem akan menambahkan
   sumber berdasarkan metadata asli.
9. Untuk pertanyaan lanjutan, gunakan riwayat percakapan untuk
   memahami maksud pengguna. Namun, gunakan konteks materi untuk
   memverifikasi fakta dan sitasi.

KONTEKS MATERI:
{context}

PERTANYAAN PENGGUNA:
{question}
""".strip()

    # Jangan tambahkan pertanyaan terbaru ke history sebelum
    # membuat messages, supaya pertanyaan tidak terkirim dua kali.
    if chat_history:
        messages = chat_history.get_messages(SYSTEM_PROMPT)
        messages.append({
            "role": "user",
            "content": (
                "Gunakan konteks berikut untuk menjawab pertanyaan.\n\n"
                + prompt
            ),
        })
    else:
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

    # Generation
    start_generation = time.time()
    full_response = ""

    try:
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=True,
        )

        for chunk in stream:
            if not chunk.choices:
                continue

            token = chunk.choices[0].delta.content

            if token:
                full_response += token
                yield token
        generation_ms = (time.time() - start_generation) * 1000

    except Exception as exc:
        print(
            f"[Generation error] {type(exc).__name__}: {exc}"
        )

        yield (
            "\n\nMaaf, terjadi kesalahan saat menghasilkan jawaban. "
            "Periksa log server untuk detailnya."
        )
        return

    # Citations: hanya tampilkan sumber jika model memberi sitasi valid.
    citation_numbers = _extract_valid_citations(
        full_response,
        len(documents),
    )

    citations_text = ""

    if (
        citation_numbers
        and not _answer_says_not_found(full_response)
    ):
        citations_text = _format_citations(
            citation_numbers,
            metadatas,
        )

    if citations_text:
        yield "\n\n" + citations_text

    elif _answer_says_not_found(full_response):
        # Jangan tampilkan kandidat retrieval sebagai bukti
        # jika model menyatakan jawabannya tidak ada di PDF.
        pass

    else:
        print(
            "[Citation warning] AI tidak menghasilkan sitasi valid; "
            "tidak menampilkan sumber fallback."
        )

        yield (
            "\n\n**Sumber materi:** "
            "Tidak ada sitasi valid yang dihasilkan."
        )

    # Simpan pertanyaan dan jawaban satu kali setelah generation selesai.
    if chat_history:
        chat_history.add_user(question)
        chat_history.add_assistant(full_response)

    # Save cache
    try:
        response_to_cache = full_response

        if citations_text:
            response_to_cache += "\n\n" + citations_text
            response_to_cache += f"\n<!-- latency: retrieval={retrieval_ms:.1f}ms generation={generation_ms:.1f}ms -->"

        save_to_cache(
            question,
            response_to_cache,
            context_key=context_key,
        )

        print("[Cache saved]")

    except Exception as exc:
        print(f"[Cache save error] {exc}")
