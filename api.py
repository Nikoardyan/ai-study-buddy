import hmac
import logging
import re
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.loader import load_document
from src.indexer import (
    index_document,
    get_collection_stats,
    detect_subject,
    collection,
)
from src.generator import generate_answer_streaming
from src.memory import SessionManager
from src.config import AUTH_TOKENS, CORS_ALLOWED_ORIGINS, DOCS_FOLDER
from src.cache_manager import invalidate_document_cache


# ============== SETUP ==============

app = FastAPI(
    title="AI Study Buddy API",
    description="RAG-based AI Tutor untuk siswa/mahasiswa dengan isolasi pengguna & sesi",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
)

UPLOAD_DIR = Path(DOCS_FOLDER)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Batasan ukuran file: 15 MB
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024

# Session Manager mengelola memori per-sesi (bukan singleton global bersama)
session_manager = SessionManager(ttl_seconds=86400, max_turns=10)
logger = logging.getLogger(__name__)

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
MAX_QUESTION_LENGTH = 4_000


# ============== SCHEMAS ==============

class ChatRequest(BaseModel):
    question: str
    subject: Optional[str] = None
    grade_level: Optional[str] = None
    session_id: Optional[str] = None


# ============== HELPERS ==============

def _validate_identifier(value: str, label: str) -> str:
    """Validate IDs used in metadata, memory, and filesystem paths."""
    value = value.strip()
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise HTTPException(
            status_code=400,
            detail=(
                f"{label} tidak valid. Gunakan 1-64 karakter: "
                "huruf, angka, garis bawah, atau tanda hubung."
            ),
        )
    return value


def _authenticated_user_id(
    x_user_id: Optional[str],
    authorization: Optional[str],
) -> str:
    """Resolve a trusted user identity when bearer-token auth is enabled."""
    if AUTH_TOKENS:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Bearer token diperlukan.")

        presented_token = authorization.removeprefix("Bearer ").strip()
        for token, user_id in AUTH_TOKENS.items():
            if hmac.compare_digest(presented_token, token):
                return _validate_identifier(user_id, "user_id pada token")

        raise HTTPException(status_code=401, detail="Bearer token tidak valid.")

    # Mode demo lokal. Jangan gunakan mode ini untuk data pengguna yang sensitif.
    return _validate_identifier(x_user_id or "public", "X-User-ID")


def _resolve_ids(
    x_user_id: Optional[str] = None,
    x_session_id: Optional[str] = None,
    req_session_id: Optional[str] = None,
    authorization: Optional[str] = None,
) -> tuple[str, str]:
    """
    Ekstrak dan sanitasi user_id dan session_id dari header atau body.
    User ID defaults to 'public' if not provided (demo mode).
    Session ID di-generate secara unik jika belum ada.
    """
    user_id = _authenticated_user_id(x_user_id, authorization)
    session_id = (x_session_id or req_session_id or "").strip()
    if not session_id:
        session_id = str(uuid.uuid4())
    return user_id, _validate_identifier(session_id, "session_id")


def _get_owned_session(session_id: str, user_id: str):
    """Return a session only when it belongs to the requesting user."""
    try:
        return session_manager.get_session(session_id, owner_id=user_id)
    except PermissionError as exc:
        # Do not reveal whether a session owned by another user exists.
        raise HTTPException(status_code=404, detail="Session tidak ditemukan.") from exc


# ============== ENDPOINTS ==============

@app.get("/")
async def root():
    """Serve frontend HTML."""
    return FileResponse("static/index.html")


@app.get("/api/status")
async def api_status(
    x_user_id: Optional[str] = Header(None),
    x_session_id: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Health check + stats per-sesi/user."""
    stats = get_collection_stats()
    user_id, session_id = _resolve_ids(x_user_id, x_session_id, authorization=authorization)
    session = _get_owned_session(session_id, user_id)

    return {
        "status": "ok",
        "service": "AI Study Buddy API",
        "total_chunks": stats["total_chunks"],
        "user_id": user_id,
        "session_id": session_id,
        "history_turns": session.turn_count,
        "active_subject": session.active_subject,
        "active_sessions_total": session_manager.active_session_count,
    }


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    subject: Optional[str] = None,
    grade_level: str = "umum",
    difficulty: str = "intermediate",
    x_user_id: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Upload materi belajar dan index ke vector database dengan isolasi user_id."""
    user_id, _ = _resolve_ids(x_user_id, authorization=authorization)
    allowed = {".pdf", ".docx", ".txt", ".md"}

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Nama file tidak valid.",
        )

    filename = Path(file.filename).name
    ext = Path(filename).suffix.lower()

    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Format {ext} tidak didukung. "
                f"Gunakan: {', '.join(sorted(allowed))}"
            ),
        )

    # Simpan file di direktori spesifik user untuk isolasi filesystem
    user_upload_dir = (UPLOAD_DIR / user_id).resolve()
    upload_root = UPLOAD_DIR.resolve()
    if upload_root not in user_upload_dir.parents:
        raise HTTPException(status_code=400, detail="user_id tidak valid.")
    user_upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = user_upload_dir / filename

    try:
        # Validasi batas ukuran file (15 MB) saat streaming
        total_bytes = 0
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 64):  # 64KB per chunk
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE_BYTES:
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"Ukuran file melebihi batas maksimum 15 MB.",
                    )
                buffer.write(chunk)

        document = load_document(str(file_path))
        detected_subject = subject or detect_subject(filename)

        index_document(
            document,
            subject=detected_subject,
            grade_level=grade_level,
            difficulty=difficulty,
            user_id=user_id,
        )

        return {
            "status": "success",
            "filename": filename,
            "user_id": user_id,
            "subject": detected_subject,
            "total_pages": document["total_pages"],
            "message": f"Berhasil index {filename}",
        }

    except HTTPException:
        file_path.unlink(missing_ok=True)
        raise

    except Exception:
        file_path.unlink(missing_ok=True)
        logger.exception("Gagal memproses upload")
        raise HTTPException(
            status_code=500,
            detail="Gagal memproses dokumen. Periksa format dan isi file.",
        )

    finally:
        await file.close()


@app.get("/documents")
async def list_documents(
    x_user_id: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """List dokumen yang diizinkan untuk user ini (milik user_id atau dokumen materi public)."""
    user_id, _ = _resolve_ids(x_user_id, authorization=authorization)

    # Filter ChromaDB agar hanya mengambil dokumen user_id atau public
    if user_id == "public":
        where_filter = {"user_id": "public"}
    else:
        where_filter = {
            "$or": [
                {"user_id": user_id},
                {"user_id": "public"},
            ]
        }

    all_data = collection.get(where=where_filter)
    metadatas = all_data.get("metadatas") or []

    documents = {}
    for meta in metadatas:
        if not meta:
            continue
        filename = meta.get("filename")
        if filename and filename not in documents:
            documents[filename] = {
                "filename": filename,
                "owner": meta.get("user_id", "public"),
                "subject": meta.get("subject", "general"),
                "grade_level": meta.get("grade_level", "umum"),
                "total_pages": meta.get("total_pages", 0),
                "indexed_at": meta.get("indexed_at", "unknown"),
            }

    return {
        "user_id": user_id,
        "total_documents": len(documents),
        "total_chunks": len(all_data.get("ids", [])),
        "documents": list(documents.values()),
    }


@app.delete("/documents/{filename}")
async def delete_document(
    filename: str,
    x_user_id: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Hapus dokumen milik user_id dan bersihkan cache terkait."""
    user_id, _ = _resolve_ids(x_user_id, authorization=authorization)
    filename = Path(filename).name

    # Cek kepemilikan dokumen sebelum menghapus
    existing = collection.get(
        where={
            "$and": [
                {"filename": filename},
                {"user_id": user_id},
            ]
        }
    )

    if not existing or not existing.get("ids"):
        raise HTTPException(
            status_code=404,
            detail=f"Dokumen '{filename}' tidak ditemukan atau Anda tidak memiliki izin untuk menghapusnya.",
        )

    # Hapus dari ChromaDB
    collection.delete(
        where={
            "$and": [
                {"filename": filename},
                {"user_id": user_id},
            ]
        }
    )

    # Hapus dari filesystem
    user_upload_dir = UPLOAD_DIR / user_id
    file_path = user_upload_dir / filename
    if file_path.exists():
        file_path.unlink()
    # Jika ada di direktori legacy root
    root_file = UPLOAD_DIR / filename
    if root_file.exists() and user_id == "public":
        root_file.unlink()

    # Invalidate cache terkait dokumen ini
    invalidate_document_cache(filename)

    return {
        "status": "deleted",
        "filename": filename,
        "user_id": user_id,
        "message": f"{filename} berhasil dihapus dan cache diinvalidasi",
    }


@app.post("/chat")
async def chat(
    request: ChatRequest,
    x_user_id: Optional[str] = Header(None),
    x_session_id: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Chat dengan session history terisolasi."""
    if not request.question or len(request.question.strip()) > MAX_QUESTION_LENGTH:
        raise HTTPException(status_code=422, detail="Pertanyaan harus berisi 1-4000 karakter.")
    user_id, session_id = _resolve_ids(
        x_user_id, x_session_id, request.session_id, authorization,
    )
    session = _get_owned_session(session_id, user_id)

    full_answer = ""
    try:
        for chunk in generate_answer_streaming(
            question=request.question,
            chat_history=session,
            subject_filter=request.subject,
            user_id=user_id,
        ):
            full_answer += chunk

        return {
            "session_id": session_id,
            "user_id": user_id,
            "question": request.question,
            "answer": full_answer,
            "subject_detected": session.active_subject,
            "turn_count": session.turn_count,
        }

    except Exception:
        logger.exception("Gagal memproses chat")
        raise HTTPException(
            status_code=500,
            detail="Gagal memproses pertanyaan. Silakan coba lagi.",
        )


@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    x_user_id: Optional[str] = Header(None),
    x_session_id: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Chat streaming dengan session history terisolasi."""
    if not request.question or len(request.question.strip()) > MAX_QUESTION_LENGTH:
        raise HTTPException(status_code=422, detail="Pertanyaan harus berisi 1-4000 karakter.")
    user_id, session_id = _resolve_ids(
        x_user_id, x_session_id, request.session_id, authorization,
    )
    session = _get_owned_session(session_id, user_id)

    def generate():
        yield from generate_answer_streaming(
            question=request.question,
            chat_history=session,
            subject_filter=request.subject,
            user_id=user_id,
        )

    response = StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
    )
    response.headers["X-Session-ID"] = session_id
    return response


@app.delete("/chat/clear")
async def clear_chat(
    x_session_id: Optional[str] = Header(None),
    session_id: Optional[str] = None,
    x_user_id: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Reset riwayat chat untuk sesi tertentu."""
    target_session = (x_session_id or session_id or "").strip()
    if not target_session:
        raise HTTPException(status_code=400, detail="session_id wajib disertakan.")
    user_id = _authenticated_user_id(x_user_id, authorization)
    target_session = _validate_identifier(target_session, "session_id")

    try:
        deleted = session_manager.clear_session(target_session, owner_id=user_id)
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail="Session tidak ditemukan.") from exc

    return {
        "status": "cleared" if deleted else "not_found",
        "session_id": target_session,
        "message": f"History sesi {target_session} berhasil di-reset",
    }


@app.get("/chat/history")
async def get_history(
    x_session_id: Optional[str] = Header(None),
    session_id: Optional[str] = None,
    x_user_id: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Lihat riwayat chat untuk sesi tertentu."""
    target_session = (x_session_id or session_id or "").strip()
    if not target_session:
        return {"turn_count": 0, "messages": [], "session_id": None}

    user_id = _authenticated_user_id(x_user_id, authorization)
    target_session = _validate_identifier(target_session, "session_id")

    session = _get_owned_session(target_session, user_id)

    return {
        "session_id": target_session,
        "turn_count": session.turn_count,
        "active_subject": session.active_subject,
        "messages": [
            {
                "role": msg["role"],
                "content": (
                    msg["content"][:200] + "..."
                    if len(msg["content"]) > 200
                    else msg["content"]
                ),
            }
            for msg in session.messages
        ],
    }
