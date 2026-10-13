"""
Tests for Prioritas 4: Answer Quality, Citations, and Abstention Evaluator
Jalankan:
    python -m unittest tests/test_evaluator_quality.py
"""

import unittest
from evaluate_answers_comprehensive import calculate_citation_support, evaluate_with_judge


class TestEvaluatorQuality(unittest.TestCase):

    def test_citation_support_calculation(self):
        """Memverifikasi perhitungan rasio kecocokan sitasi terhadap halaman ground truth."""
        answer = "HCI adalah ilmu interaksi [1], dan usability sangat penting [2]."
        metas = [
            {"page_number": 3},  # chunk 1 -> Hal 3 (Primary)
            {"page_number": 7},  # chunk 2 -> Hal 7 (Primary)
            {"page_number": 15}, # chunk 3
        ]
        # Target halaman: [3, 7]
        score = calculate_citation_support(answer, metas, primary_pages=[3, 7], acceptable_pages=[])
        self.assertEqual(score, 1.0)

        # Jika chunk 2 bukan halaman target: misal target hanya [3]
        score_partial = calculate_citation_support(answer, metas, primary_pages=[3], acceptable_pages=[])
        self.assertEqual(score_partial, 0.5)

    def test_citation_support_refusal(self):
        """Jika jawaban adalah penolakan (abstention) dan tidak ada sitasi, skor sitasi harus 1.0 (valid)."""
        answer_refusal = "Maaf, informasi tentang quicksort tidak tersedia dalam materi yang diunggah."
        metas = [{"page_number": 16}, {"page_number": 21}]
        score = calculate_citation_support(answer_refusal, metas, primary_pages=[], acceptable_pages=[])
        self.assertEqual(score, 1.0)

    def test_llm_judge_output_structure(self):
        """Memverifikasi bahwa parser evaluasi LLM mengembalikan tipe data yang tepat."""
        res = evaluate_with_judge(
            question="Apa itu usability?",
            context="Usability adalah kemudahan penggunaan sistem.",
            answer="Usability adalah kemudahan penggunaan sistem.",
            reference_answer="Kemudahan sistem digunakan.",
            expected_behavior="answer"
        )
        self.assertIn("faithfulness", res)
        self.assertIn("answer_relevancy", res)
        self.assertIn("is_refusal", res)
        self.assertIsInstance(res["faithfulness"], float)
        self.assertIsInstance(res["answer_relevancy"], float)
        self.assertIsInstance(res["is_refusal"], bool)


if __name__ == "__main__":
    unittest.main()
