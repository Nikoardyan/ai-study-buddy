"""
src/evaluator.py
RAGAS Evaluation dengan Groq LLM
Level 2 Part 3 - RAG Production
"""
import os
import numpy as np
from dotenv import load_dotenv
from typing import List, Dict
from datasets import Dataset

load_dotenv()

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Setup Groq LLM untuk RAGAS judge
groq_llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,
)

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

ragas_llm = LangchainLLMWrapper(groq_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embedder)


def safe_score(value) -> float:
    """Convert list/single value ke float aman."""
    if isinstance(value, list):
        # Filter NaN, ambil average
        valid_values = [
            v for v in value 
            if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v))
        ]
        if not valid_values:
            return 0.0
        return float(np.mean(valid_values))
    try:
        f = float(value)
        return 0.0 if np.isnan(f) else f
    except (TypeError, ValueError):
        return 0.0


def prepare_eval_dataset(test_cases: List[Dict]) -> Dataset:
    """Convert test cases ke Dataset RAGAS."""
    return Dataset.from_list(test_cases)


def evaluate_rag(test_cases: List[Dict]) -> Dict:
    """Evaluate RAG dengan RAGAS pakai Groq."""
    if not test_cases:
        return {"error": "No test cases provided"}
    
    print(f"\n[📊 RAGAS Evaluation: {len(test_cases)} test cases]")
    
    dataset = prepare_eval_dataset(test_cases)
    
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
    
    try:
        print("[Running RAGAS metrics... ini bisa 1-3 menit]")
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        
        # DEBUG: Print raw result buat tau struktur
        print(f"\n[DEBUG] Raw result type: {type(result)}")
        print(f"[DEBUG] Keys: {list(result.keys()) if hasattr(result, 'keys') else 'N/A'}")
        
        scores = {
            "faithfulness": round(safe_score(result["faithfulness"]), 3),
            "answer_relevancy": round(safe_score(result["answer_relevancy"]), 3),
            "context_precision": round(safe_score(result["context_precision"]), 3),
            "context_recall": round(safe_score(result["context_recall"]), 3),
            "num_test_cases": len(test_cases)
        }
        
        return scores
    
    except Exception as e:
        import traceback
        print(f"⚠️ Evaluation error: {e}")
        print(f"[Traceback]")
        traceback.print_exc()
        return {"error": str(e)}


def print_eval_report(scores: Dict):
    """Print evaluation report."""
    if "error" in scores:
        print(f"\n❌ Evaluation Failed: {scores['error']}")
        return
    
    print("\n" + "="*50)
    print("📊 RAG EVALUATION REPORT (RAGAS)")
    print("="*50)
    
    metrics_info = {
        "faithfulness": ("Faithfulness", "ngarang ga (target > 0.85)"),
        "answer_relevancy": ("Answer Relevancy", "jawab pertanyaan (target > 0.85)"),
        "context_precision": ("Context Precision", "retrieval quality (target > 0.80)"),
        "context_recall": ("Context Recall", "kelengkapan info (target > 0.75)"),
    }
    
    for key, (label, desc) in metrics_info.items():
        score = scores.get(key, 0)
        emoji = "✅" if score >= 0.80 else "⚠️" if score >= 0.65 else "❌"
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"\n{emoji} {label}: {score:.3f}")
        print(f"   [{bar}] ({desc})")
    
    print("\n" + "="*50)
    print(f"Tested with: {scores['num_test_cases']} queries")
    print("="*50 + "\n")


def test_evaluator():
    """Quick test."""
    test_cases = [
        {
            "question": "Apa itu turunan?",
            "answer": "Turunan adalah laju perubahan suatu fungsi. Turunan x² adalah 2x.",
            "contexts": [
                "Turunan adalah konsep matematika yang mengukur laju perubahan suatu fungsi.",
                "Turunan dari x² adalah 2x."
            ],
            "ground_truth": "Turunan adalah laju perubahan suatu fungsi. Turunan x² = 2x."
        },
        {
            "question": "Berapa nilai turunan x²?",
            "answer": "Turunan dari x² adalah 2x.",
            "contexts": [
                "Aturan turunan: d/dx(x^n) = n*x^(n-1)",
                "Untuk x²: d/dx(x²) = 2x"
            ],
            "ground_truth": "Turunan x² adalah 2x."
        }
    ]
    
    print("\n🧪 Testing RAGAS Evaluator (with Groq LLM)...")
    scores = evaluate_rag(test_cases)
    print_eval_report(scores)


if __name__ == "__main__":
    test_evaluator()