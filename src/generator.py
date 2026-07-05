from openai import OpenAI
from src.config import (
    API_KEY, BASE_URL, LLM_MODEL,
    TEMPERATURE, MAX_TOKENS, TOP_K, SYSTEM_PROMPT
)
from src.indexer import collection
from src.memory import ChatHistory
from src.reranker import rerank_documents
from src.hybrid_retriever import hybrid_retrieve
from src.cache_manager import check_cache, save_to_cache  # ← NEW!
from langchain_core.documents import Document


# Setup LLM client
llm = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def detect_subject_from_question(question: str) -> str | None:
    """Detect subject secara dinamis menggunakan LLM."""
    try:
        response = llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "Extract the main educational subject from the user's question (e.g. Matematika, Fisika, Biologi, Sejarah). Only reply with the subject name in single word. If no specific subject is detected, reply with 'umum'."},
                {"role": "user", "content": question}
            ],
            temperature=0.0,
            max_tokens=10
        )
        subject = response.choices[0].message.content.strip().lower()
        return subject if subject != "umum" and subject != "" else None
    except Exception as e:
        print(f"[Error detecting subject] {e}")
        return None


def retrieve(question: str, top_k: int = None, where_filter: dict = None) -> tuple:
    """Vector search basic (legacy)."""
    top_k = top_k or TOP_K
    query_args = {"query_texts": [question], "n_results": top_k}
    if where_filter:
        query_args["where"] = where_filter

    results = collection.query(**query_args)
    chunks = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    return chunks, metadatas


def retrieve_with_rerank(
    question: str,
    initial_k: int = 20,
    final_n: int = 5,
    where_filter: dict = None
) -> tuple:
    """
    PRODUCTION-GRADE retrieval:
    1. Hybrid Search (Vector + BM25 + RRF)
    2. Cohere Reranking
    """
    # Step 1: Hybrid Search
    chunks, metadatas = hybrid_retrieve(
        question=question,
        top_k=initial_k,
        where_filter=where_filter
    )

    if len(chunks) <= final_n:
        return chunks, metadatas

    # Step 2: Reranking
    documents = [
        Document(page_content=chunk, metadata=meta)
        for chunk, meta in zip(chunks, metadatas)
    ]

    reranked_docs = rerank_documents(
        query=question,
        documents=documents,
        top_n=final_n
    )

    final_chunks = [doc.page_content for doc in reranked_docs]
    final_metadatas = [doc.metadata for doc in reranked_docs]

    print(f"[✨ Reranked: {len(chunks)} → {len(final_chunks)} chunks]")
    return final_chunks, final_metadatas


def rewrite_query(question: str, chat_history: ChatHistory) -> str:
    """Rewrite follow-up question into a standalone query using chat history."""
    if not chat_history.messages:
        return question
    
    try:
        # Ambil sedikit history untuk konteks (misal 2 pasang terakhir)
        history_text = ""
        for msg in chat_history.messages[-4:]:
            role = "Siswa" if msg["role"] == "user" else "Tutor"
            # Hindari memasukkan context besar dari prompt sebelumnya
            content = msg['content'].split("PERTANYAAN SISWA:")[-1].strip() if "PERTANYAAN SISWA:" in msg['content'] else msg['content']
            history_text += f"{role}: {content}\n"
            
        prompt = f"""Diberikan riwayat percakapan berikut dan pertanyaan lanjutan dari siswa. 
Tulis ulang pertanyaan lanjutan menjadi pertanyaan utuh (standalone query) yang mengandung subjek dan konteks penuh agar bisa dicari di database.
HANYA berikan pertanyaan yang sudah ditulis ulang, tanpa pembukaan/penutup.

Riwayat Percakapan:
{history_text}

Pertanyaan Lanjutan Siswa: {question}

Pertanyaan Utuh:"""

        response = llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=60
        )
        rewritten = response.choices[0].message.content.strip()
        print(f"\n[Query Rewritten] '{question}' -> '{rewritten}'")
        return rewritten
    except Exception as e:
        print(f"[Error rewriting query] {e}")
        return question


def generate_answer_streaming(
    question: str,
    chat_history: ChatHistory,
    subject_filter: str = None,
    filename_filter: str = None
):
    """
    Production RAG pipeline:
    - LLM Subject detection
    - Cache check with context
    - Query Rewriting (Contextual Memory)
    - Hybrid Search
    - Reranking
    - LLM generation
    - Cache save with context
    """
    
    # Auto-detect subject if not explicitly filtered
    detected_subject = subject_filter or detect_subject_from_question(question)

    where_filter = {}
    if detected_subject:
        where_filter["subject"] = detected_subject
        print(f"\n[Filter: subject = {detected_subject}]")
    if filename_filter:
        where_filter["filename"] = filename_filter
        print(f"[Filter: filename = {filename_filter}]")
        
    context_key = str(where_filter)

    # ═══════════════════════════════════════════════════════
    # STEP 0: CHECK CACHE DULU (FAST PATH!)
    # ═══════════════════════════════════════════════════════
    cached_response = check_cache(question, context_key=context_key)
    if cached_response:
        # Cache HIT! Return cached, skip LLM
        chat_history.add_user(question)
        chat_history.add_assistant(cached_response)
        yield cached_response
        return

    # ═══════════════════════════════════════════════════════
    # STEP 1: QUERY REWRITING (FOR CONVERSATIONAL MEMORY)
    # ═══════════════════════════════════════════════════════
    search_query = rewrite_query(question, chat_history)

    # ═══════════════════════════════════════════════════════
    # STEP 2-7: NORMAL RAG FLOW (kalau cache MISS)
    # ═══════════════════════════════════════════════════════

    # Hybrid + Rerank
    chunks, metadatas = retrieve_with_rerank(
        question=search_query,
        initial_k=20,
        final_n=5,
        where_filter=where_filter if where_filter else None
    )

    # Fallback
    if not chunks and where_filter:
        print(f"[Ga ada hasil dengan filter, coba tanpa filter...]")
        chunks, metadatas = retrieve_with_rerank(
            question=search_query, initial_k=20, final_n=5
        )

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

    user_prompt_with_context = f"""MATERI BELAJAR (CONTEXT):
{context}

PERTANYAAN SISWA: {question}

INSTRUKSI KHUSUS: 
1. Jika siswa hanya menyapa (Halo, perkenalan diri, salam) atau obrolan santai, ABAIKAN materi belajar, jawablah sapaannya secara ramah dan natural tanpa kutipan (citation).
2. Jika siswa bertanya/membahas materi, jawablah sebagai AI Tutor dan WAJIB gunakan kutipan (contoh: [1], [2]) berdasarkan MATERI BELAJAR di atas."""

    chat_history.add_user(user_prompt_with_context)
    messages = chat_history.get_messages(SYSTEM_PROMPT)

    # Stream response
    stream = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=True
    )

    full_response = ""

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            full_response += content
            yield content

    # Add citations only if the LLM actually used them in the response
    citations_used = any(f"[{i}]" in full_response for i in range(1, len(metadatas) + 1))
    
    citations_text = ""
    if citations_used and metadatas:
        citations_text = "\n\nSumber Materi:\n"
        for i, meta in enumerate(metadatas, 1):
            subject_info = meta.get("subject", "general")
            citations_text += (f"[{i}] {meta.get('filename', 'unknown')}, "
                              f"halaman {meta.get('page_number', 'N/A')} "
                              f"(subject: {subject_info})\n")
        yield citations_text

    chat_history.add_assistant(full_response)
    
    # ═══════════════════════════════════════════════════════
    # STEP 8: SAVE TO CACHE (untuk query berikutnya!)
    # ═══════════════════════════════════════════════════════
    full_with_citations = full_response + citations_text
    save_to_cache(question, full_with_citations, context_key=context_key)
    print(f"[💾 Saved to cache]")