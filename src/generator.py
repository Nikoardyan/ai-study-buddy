"""
Generator - generate jawaban dengan citation.
Update TASK 3: Tambah conversation memory.
"""
from openai import OpenAI
from src.config import (
    API_KEY, BASE_URL, LLM_MODEL,
    TEMPERATURE, MAX_TOKENS, TOP_K, SYSTEM_PROMPT
)
from src.indexer import collection
from src.memory import ChatHistory   # ← TAMBAH IMPORT INI


# Setup LLM client
llm = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def detect_subject_from_question(question: str) -> str | None:
    """Detect subject dari pertanyaan user."""
    question_lower = question.lower()

    subject_keywords = {
        "matematika": ["matematika", "math", "kalkulus",
                       "turunan", "integral", "aljabar"],
        "fisika":     ["fisika", "physics", "newton",
                       "gaya", "kecepatan", "energi"],
        "kimia":      ["kimia", "chemistry", "reaksi kimia"],
        "biologi":    ["biologi", "biology", "sel"],
        "python":     ["python", "programming", "coding",
                       "pemrograman", "loop", "variabel python"],
    }

    for subject, keywords in subject_keywords.items():
        for keyword in keywords:
            if keyword in question_lower:
                return subject

    return None


def retrieve(
    question: str,
    top_k: int = None,
    where_filter: dict = None
) -> tuple:
    """Retrieve chunks relevan dari vector DB."""
    top_k = top_k or TOP_K

    query_args = {
        "query_texts": [question],
        "n_results": top_k
    }

    if where_filter:
        query_args["where"] = where_filter

    results = collection.query(**query_args)

    chunks = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []

    return chunks, metadatas


def generate_answer_streaming(
    question: str,
    chat_history: ChatHistory,     # ← TAMBAH PARAMETER INI
    subject: str = None,
    grade_level: str = None
):
    """
    Streaming version dengan conversation memory.
    
    Args:
        question: Pertanyaan user
        chat_history: Object ChatHistory yang nyimpen history
        subject: Filter subject (optional)
        grade_level: Filter grade (optional)
    """

    # Auto-detect subject
    detected_subject = subject or detect_subject_from_question(question)

    # Build filter
    where_filter = {}
    if detected_subject:
        where_filter["subject"] = detected_subject
        print(f"\n[Filter: subject = {detected_subject}]")
    if grade_level:
        where_filter["grade_level"] = grade_level

    # Retrieve relevant chunks
    chunks, metadatas = retrieve(
        question=question,
        where_filter=where_filter if where_filter else None
    )

    # Fallback kalau ga ada hasil
    if not chunks and where_filter:
        print(f"[Ga ada hasil dengan filter, coba tanpa filter...]")
        chunks, metadatas = retrieve(question=question)

    if not chunks:
        no_result = ("Hmm, belum ada materi terkait. "
                    "Coba upload materi yang relevan dulu ya!")
        chat_history.add_user(question)
        chat_history.add_assistant(no_result)
        yield no_result
        return

    # Format context
    context_parts = [f"[{i+1}] {chunk}" for i, chunk in enumerate(chunks)]
    context = "\n\n".join(context_parts)

    # User prompt dengan context
    user_prompt_with_context = f"""MATERI BELAJAR (CONTEXT):
{context}

PERTANYAAN SISWA: {question}

Jawab dengan gaya AI Tutor. WAJIB pakai citation [1], [2], dst."""

    # ============================================
    # INI BEDANYA DARI TASK 2:
    # Pakai chat_history.get_messages() bukan hardcode messages
    # ============================================

    # Tambah pertanyaan user ke history dulu
    chat_history.add_user(user_prompt_with_context)

    # Build messages dengan history
    messages = chat_history.get_messages(SYSTEM_PROMPT)

    # Stream response
    stream = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,       # ← pakai history, bukan hardcode
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=True
    )

    # Kumpulkan full response untuk disimpan ke history
    full_response = ""

    # Yield jawaban kata per kata
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            full_response += content
            yield content

    # Yield citations
    citations_text = "\n\nSumber Materi:\n"
    for i, meta in enumerate(metadatas, 1):
        subject_info = meta.get("subject", "general")
        citations_text += (f"[{i}] {meta.get('filename', 'unknown')}, "
                          f"halaman {meta.get('page_number', 'N/A')} "
                          f"(subject: {subject_info})\n")

    yield citations_text

    # Simpan full response ke history (TANPA context PDF)
    # Kenapa tanpa context? Biar history ga terlalu panjang
    chat_history.add_assistant(full_response)