"""
Tests for Multi-User & Multi-Session Isolation (C-01 & C-02)
Jalankan:
    python -m unittest tests/test_isolation.py
"""

import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from api import app, session_manager
from src.memory import SessionManager
from src.indexer import collection, index_document


class TestIsolation(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_session_manager_isolation(self):
        """Memverifikasi bahwa dua session_id tidak saling berbagi percakapan."""
        sm = SessionManager(ttl_seconds=3600, max_turns=5)
        sess_a = sm.get_session("user_a_session")
        sess_b = sm.get_session("user_b_session")

        sess_a.add_user("Halo dari User A")
        sess_a.add_assistant("Halo A, ada yang bisa dibantu?")

        sess_b.add_user("Halo dari User B")

        # Riwayat sess_a harus 2 pesan, sess_b harus 1 pesan
        self.assertEqual(len(sess_a.messages), 2)
        self.assertEqual(len(sess_b.messages), 1)
        self.assertIn("User A", sess_a.messages[0]["content"])
        self.assertIn("User B", sess_b.messages[0]["content"])

    def test_api_session_history_isolation(self):
        """Endpoint /chat/history harus mengembalikan riwayat yang terpisah per session_id."""
        sess_1 = "test-session-1"
        sess_2 = "test-session-2"

        # Simpan state di session 1
        s1 = session_manager.get_session(sess_1)
        s1.clear()
        s1.add_user("Pesan rahasia sesi 1")

        # Cek history session 2 (harus kosong)
        res2 = self.client.get("/chat/history", headers={"X-Session-ID": sess_2})
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.json()["messages"], [])

        # Cek history session 1 (harus ada pesan rahasia)
        res1 = self.client.get("/chat/history", headers={"X-Session-ID": sess_1})
        self.assertEqual(res1.status_code, 200)
        self.assertEqual(len(res1.json()["messages"]), 1)
        self.assertIn("rahasia sesi 1", res1.json()["messages"][0]["content"])

        # Reset session 1
        del_res = self.client.delete("/chat/clear", headers={"X-Session-ID": sess_1})
        self.assertEqual(del_res.status_code, 200)
        self.assertEqual(del_res.json()["status"], "cleared")

    def test_document_owner_isolation_and_unauthorized_deletion(self):
        """User B tidak boleh menghapus dokumen milik User A."""
        # Index dummy doc milik user_alpha
        doc_data = {
            "filename": "alpha_private_notes.txt",
            "format": "txt",
            "pages": [{"page_number": 1, "content": "Catatan rahasia alpha."}],
            "total_pages": 1,
        }
        index_document(doc_data, user_id="user_alpha")

        # User beta mencoba menghapus dokumen milik user_alpha -> harus 404 (Not Found / Unauthorized)
        res_del = self.client.delete(
            "/documents/alpha_private_notes.txt",
            headers={"X-User-ID": "user_beta"}
        )
        self.assertEqual(res_del.status_code, 404)

        # User alpha menghapus dokumennya sendiri -> harus 200 (Success)
        res_del_alpha = self.client.delete(
            "/documents/alpha_private_notes.txt",
            headers={"X-User-ID": "user_alpha"}
        )
        self.assertEqual(res_del_alpha.status_code, 200)
        self.assertEqual(res_del_alpha.json()["status"], "deleted")

    def test_document_list_isolation(self):
        """User hanya melihat dokumen miliknya dan dokumen public."""
        doc_x = {
            "filename": "doc_x_private.txt",
            "format": "txt",
            "pages": [{"page_number": 1, "content": "Dokumen X"}],
            "total_pages": 1,
        }
        index_document(doc_x, user_id="user_x")

        # User Y request /documents -> doc_x_private.txt tidak boleh muncul
        res_y = self.client.get("/documents", headers={"X-User-ID": "user_y"})
        self.assertEqual(res_y.status_code, 200)
        filenames_y = [d["filename"] for d in res_y.json()["documents"]]
        self.assertNotIn("doc_x_private.txt", filenames_y)

        # Cleanup
        collection.delete(where={"filename": "doc_x_private.txt"})

    def test_rejects_unsafe_identifiers(self):
        """ID dari header tidak boleh menjadi path traversal atau metadata liar."""
        unsafe_user = self.client.get(
            "/documents",
            headers={"X-User-ID": "../../outside"},
        )
        self.assertEqual(unsafe_user.status_code, 400)

        unsafe_session = self.client.get(
            "/api/status",
            headers={"X-Session-ID": "../../other-session"},
        )
        self.assertEqual(unsafe_session.status_code, 400)

    def test_session_cannot_be_read_or_cleared_by_another_user(self):
        """Sesi yang sudah terikat ke user A tidak dapat dipakai oleh user B."""
        session_id = "owned-session-1"
        owner_headers = {
            "X-User-ID": "owner-alpha",
            "X-Session-ID": session_id,
        }
        other_headers = {
            "X-User-ID": "owner-beta",
            "X-Session-ID": session_id,
        }

        created = self.client.get("/api/status", headers=owner_headers)
        self.assertEqual(created.status_code, 200)

        history = self.client.get("/chat/history", headers=other_headers)
        self.assertEqual(history.status_code, 404)

        clear = self.client.delete("/chat/clear", headers=other_headers)
        self.assertEqual(clear.status_code, 404)

        owner_clear = self.client.delete("/chat/clear", headers=owner_headers)
        self.assertEqual(owner_clear.status_code, 200)

    def test_bearer_token_identity_overrides_user_header(self):
        """Pada mode deploy, user ID harus berasal dari bearer token, bukan header."""
        with patch("api.AUTH_TOKENS", {"secret-test-token": "token-user"}):
            no_token = self.client.get("/documents")
            self.assertEqual(no_token.status_code, 401)

            response = self.client.get(
                "/documents",
                headers={
                    "Authorization": "Bearer secret-test-token",
                    "X-User-ID": "spoofed-user",
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["user_id"], "token-user")


if __name__ == "__main__":
    unittest.main()
