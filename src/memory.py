"""
Conversation Memory untuk AI Study Buddy.
Update: Tambah active_subject tracking.
"""


class ChatHistory:
    """
    Simpan history percakapan + subject aktif.
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.messages = []
        self.active_subject = None    # ← FIX: tambah ini

    def add_user(self, message: str):
        """Tambah pesan dari user."""
        self.messages.append({
            "role": "user",
            "content": message
        })
        self._trim()

    def add_assistant(self, message: str):
        """Tambah pesan dari AI."""
        self.messages.append({
            "role": "assistant",
            "content": message
        })

    def set_subject(self, subject: str):
        """Simpan subject aktif."""
        if subject and subject != self.active_subject:
            print(f"[Subject aktif: {subject}]")
            self.active_subject = subject

    def get_messages(self, system_prompt: str) -> list:
        """Return semua messages siap dikirim ke LLM."""
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
        self.active_subject = None    # ← FIX: reset subject juga
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