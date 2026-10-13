import time
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

    def get_session(self, session_id: str) -> ChatHistory:
        """Ambil atau buat sesi percakapan untuk session_id tertentu."""
        self._cleanup()
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatHistory(max_turns=self.max_turns)
        session = self._sessions[session_id]
        session.last_accessed = time.time()
        return session

    def clear_session(self, session_id: str) -> bool:
        """Hapus riwayat sesi tertentu."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def session_exists(self, session_id: str) -> bool:
        """Periksa apakah session_id aktif."""
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

    @property
    def active_session_count(self) -> int:
        return len(self._sessions)