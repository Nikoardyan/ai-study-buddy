import time
from threading import RLock
from typing import Dict, Optional


class ChatHistory:
    """
    Simpan history percakapan + subject aktif untuk satu sesi.
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.messages = []
        self.active_subject = None
        self.last_accessed = time.time()

    def add_user(self, message: str):
        """Tambah pesan dari user."""
        self.messages.append({
            "role": "user",
            "content": message
        })
        self._trim()
        self.last_accessed = time.time()

    def add_assistant(self, message: str):
        """Tambah pesan dari AI."""
        self.messages.append({
            "role": "assistant",
            "content": message
        })
        self.last_accessed = time.time()

    def set_subject(self, subject: str):
        """Simpan subject aktif."""
        if subject and subject != self.active_subject:
            print(f"[Subject aktif: {subject}]")
            self.active_subject = subject
        self.last_accessed = time.time()

    def get_messages(self, system_prompt: str) -> list:
        """Return semua messages siap dikirim ke LLM."""
        self.last_accessed = time.time()
        return [
            {"role": "system", "content": system_prompt}
        ] + self.messages

    def _trim(self):
        """Batasi history biar ga overflow."""
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[2:]

    def clear(self):
        """Reset history + subject."""
        self.messages = []
        self.active_subject = None
        self.last_accessed = time.time()
        print("[History + subject cleared]")

    def show(self):
        """Print history untuk debugging."""
        print("\n=== CHAT HISTORY ===")
        print(f"Subject aktif: {self.active_subject or 'None'}")
        print(f"Total turns: {self.turn_count}")
        print("---")
        for i, msg in enumerate(self.messages, 1):
            role = msg["role"].upper()
            content = msg["content"][:80].replace("\n", " ")
            print(f"{i}. [{role}]: {content}...")
        print("===================\n")

    @property
    def turn_count(self) -> int:
        """Berapa banyak giliran percakapan."""
        return len(self.messages) // 2


class SessionManager:
    """
    Mengelola instance ChatHistory yang terisolasi per session_id.
    Mendukung pembersihan sesi yang sudah kedaluwarsa (TTL).
    """

    def __init__(self, ttl_seconds: int = 86400, max_turns: int = 10):
        self.ttl_seconds = ttl_seconds
        self.max_turns = max_turns
        self._sessions: Dict[str, ChatHistory] = {}
        self._owners: Dict[str, str] = {}
        self._lock = RLock()

    def get_session(
        self,
        session_id: str,
        owner_id: Optional[str] = None,
    ) -> ChatHistory:
        """Ambil sesi dan pastikan sesi hanya digunakan pemilik yang sama."""
        with self._lock:
            self._cleanup()
            stored_owner = self._owners.get(session_id)

            if stored_owner and owner_id and stored_owner != owner_id:
                raise PermissionError("Session bukan milik user ini.")

            if session_id not in self._sessions:
                self._sessions[session_id] = ChatHistory(max_turns=self.max_turns)

            # Sesi lama yang belum memiliki owner (misalnya state dari versi lama)
            # akan diklaim oleh user pertama yang mengaksesnya.
            if owner_id and not stored_owner:
                self._owners[session_id] = owner_id

            session = self._sessions[session_id]
            session.last_accessed = time.time()
            return session

    def clear_session(self, session_id: str, owner_id: Optional[str] = None) -> bool:
        """Hapus riwayat sesi; pemilik lain tidak boleh menghapusnya."""
        with self._lock:
            stored_owner = self._owners.get(session_id)
            if stored_owner and owner_id and stored_owner != owner_id:
                raise PermissionError("Session bukan milik user ini.")
            if session_id in self._sessions:
                del self._sessions[session_id]
                self._owners.pop(session_id, None)
                return True
            return False

    def session_exists(self, session_id: str) -> bool:
        """Periksa apakah session_id aktif."""
        with self._lock:
            return session_id in self._sessions

    def _cleanup(self):
        """Bersihkan sesi yang tidak aktif melewati batas TTL."""
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if (now - s.last_accessed) > self.ttl_seconds
        ]
        for sid in expired:
            del self._sessions[sid]
            self._owners.pop(sid, None)

    @property
    def active_session_count(self) -> int:
        with self._lock:
            return len(self._sessions)
