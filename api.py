"""
AI Study Buddy - FastAPI Version
TASK 4 + 5: API + Frontend
"""
import os
import shutil
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from src.loader import load_document
from src.indexer import (
    index_document,
    get_collection_stats,
    detect_subject,
    collection
)
from src.generator import generate_answer_streaming
from src.memory import ChatHistory
from src.config import DOCS_FOLDER


# ============== SETUP ==============
app = FastAPI(
    title="AI Study Buddy API",
    description="RAG-based AI Tutor untuk siswa/mahasiswa",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Folder upload
UPLOAD_DIR = Path(DOCS_FOLDER)
UPLOAD_DIR.mkdir(exist_ok=True)

# Global chat history
chat_history = ChatHistory(max_turns=10)


# ============== SCHEMAS ==============

class ChatRequest(BaseModel):
    question: str
    subject: Optional[str] = None
    grade_level: Optional[str] = None


# ============== ENDPOINTS ==============

@app.get("/")
async def root():
    """Serve frontend HTML."""
    return FileResponse("static/index.html")


@app.get("/api/status")
async def api_status():
    """Health check + stats."""
    stats = get_collection_stats()
    return {
        "status": "ok",
        "service": "AI Study Buddy API",
        "total_chunks": stats["total_chunks"],
        "history_turns": chat_history.turn_count,
        "active_subject": chat_history.active_subject
    }


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    subject: Optional[str] = None,
    grade_level: str = "umum",
    difficulty: str = "intermediate"
):
    """Upload PDF/DOCX materi belajar. Auto-index ke vector DB."""
    # Validasi format
    allowed = {".pdf", ".docx", ".txt", ".md"}
    ext = Path(file.filename).suffix.lower()

    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Format {ext} tidak di-support. Gunakan: {', '.join(allowed)}"
        )

    # Simpan file
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load & index
    try:
        document = load_document(str(file_path))
        detected_subject = subject or detect_subject(file.filename)

        index_document(
            document,
            subject=detected_subject,
            grade_level=grade_level,
            difficulty=difficulty
        )

        return {
            "status": "success",
            "filename": file.filename,
            "subject": detected_subject,
            "total_pages": document["total_pages"],
            "message": f"Berhasil index {file.filename}"
        }

    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Error indexing: {str(e)}")


@app.get("/documents")
async def list_documents():
    """List semua dokumen yang udah di-index."""
    all_data = collection.get()
    metadatas = all_data.get("metadatas", [])

    documents = {}
    for meta in metadatas:
        filename = meta.get("filename")
        if filename and filename not in documents:
            documents[filename] = {
                "filename": filename,
                "subject": meta.get("subject", "general"),
                "grade_level": meta.get("grade_level", "umum"),
                "total_pages": meta.get("total_pages", 0),
                "indexed_at": meta.get("indexed_at", "unknown"),
            }

    return {
        "total_documents": len(documents),
        "total_chunks": get_collection_stats()["total_chunks"],
        "documents": list(documents.values())
    }


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Hapus dokumen dari index dan disk."""
    collection.delete(where={"filename": filename})

    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        file_path.unlink()

    return {
        "status": "deleted",
        "filename": filename,
        "message": f"{filename} berhasil dihapus"
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat sync - tunggu sampai jawaban lengkap."""
    full_answer = ""

    for word in generate_answer_streaming(
        question=request.question,
        chat_history=chat_history,
        subject=request.subject,
        grade_level=request.grade_level
    ):
        full_answer += word

    return {
        "question": request.question,
        "answer": full_answer,
        "subject_detected": chat_history.active_subject,
        "turn_count": chat_history.turn_count
    }


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Chat streaming - jawaban muncul kata per kata."""
    def generate():
        for word in generate_answer_streaming(
            question=request.question,
            chat_history=chat_history,
            subject=request.subject,
            grade_level=request.grade_level
        ):
            yield word

    return StreamingResponse(generate(), media_type="text/plain")


@app.delete("/chat/clear")
async def clear_chat():
    """Reset conversation history."""
    chat_history.clear()
    return {
        "status": "cleared",
        "message": "History chat berhasil di-reset"
    }


@app.get("/chat/history")
async def get_history():
    """Lihat history chat."""
    return {
        "turn_count": chat_history.turn_count,
        "active_subject": chat_history.active_subject,
        "messages": [
            {
                "role": msg["role"],
                "content": msg["content"][:200] + "..."
                if len(msg["content"]) > 200
                else msg["content"]
            }
            for msg in chat_history.messages
        ]
    }