"""
Regression & Unit Tests untuk AI Study Buddy
Memverifikasi metrik, isolasi, tokenisasi, dan helper function.
Jalankan:
    python -m unittest tests/test_rag_pipeline.py
"""

import unittest
from src.hybrid_retriever import tokenize, reciprocal_rank_fusion
from src.generator import _extract_valid_citations, _answer_says_not_found
from evaluate_rag import score_item


class TestRAGPipeline(unittest.TestCase):

    def test_tokenize(self):
        """Uji tokenisasi mempertahankan istilah teknis dan angka."""
        tokens = tokenize("HCI dan UX di tahun 2026")
        self.assertEqual(tokens, ["hci", "dan", "ux", "di", "tahun", "2026"])

    def test_reciprocal_rank_fusion(self):
        """Uji ranking RRF dengan bobot."""
        vec = ["doc1", "doc2", "doc3"]
        bm25 = ["doc2", "doc1", "doc4"]
        fused = reciprocal_rank_fusion(vec, bm25, k=60, vector_weight=0.6, bm25_weight=0.4)
        doc_ids = [doc_id for doc_id, score in fused]
        # doc2 dan doc1 muncul di kedua list, harus berada di posisi teratas
        self.assertIn(doc_ids[0], ["doc1", "doc2"])
        self.assertIn(doc_ids[1], ["doc1", "doc2"])
        self.assertEqual(len(doc_ids), 4)

    def test_extract_valid_citations(self):
        """Uji ekstraksi sitasi valid dari teks model."""
        text = "IMK adalah ilmu interaksi manusia komputer [1], dan usability penting [2]."
        citations = _extract_valid_citations(text, total_chunks=3)
        self.assertEqual(citations, [1, 2])

        # Sitasi di luar batas total_chunks harus diabaikan
        text_out_of_bounds = "Klaim tanpa dasar [99]."
        citations_invalid = _extract_valid_citations(text_out_of_bounds, total_chunks=3)
        self.assertEqual(citations_invalid, [])

    def test_answer_says_not_found(self):
        """Uji deteksi penolakan / abstention."""
        ans_refused = "Maaf, informasi ini tidak ditemukan dalam materi yang diunggah."
        self.assertTrue(_answer_says_not_found(ans_refused))

        ans_answered = "Usability adalah tingkat kemudahan sistem digunakan."
        self.assertFalse(_answer_says_not_found(ans_answered))

    def test_score_item_metrics(self):
        """Uji kebenaran rumus Hit@k, MRR@k, dan Recall@k."""
        item = {
            "primary": {3, 4},
            "acceptable": {19},
        }
        # Hit ada di rank 2 (halaman 3)
        hits = [
            {"page": 10},
            {"page": 3},
            {"page": 15},
            {"page": 20},
            {"page": 21},
        ]
        res = score_item(hits, item)
        self.assertEqual(res["hit"], 1)
        self.assertEqual(res["rank"], 2)
        self.assertAlmostEqual(res["rr"], 0.5)
        self.assertAlmostEqual(res["recall"], 0.5)  # 1 dari 2 halaman primer (halaman 3)
        self.assertEqual(res["lenient_hit"], 1)

    def test_score_item_miss(self):
        """Uji kondisi MISS pada metrik."""
        item = {
            "primary": {3},
            "acceptable": set(),
        }
        hits = [
            {"page": 10},
            {"page": 12},
            {"page": 15},
        ]
        res = score_item(hits, item)
        self.assertEqual(res["hit"], 0)
        self.assertEqual(res["rank"], "")
        self.assertEqual(res["rr"], 0.0)
        self.assertEqual(res["recall"], 0.0)
        self.assertEqual(res["lenient_hit"], 0)


if __name__ == "__main__":
    unittest.main()
