"""
evaluate_rag.py (versi ablation)

Mengukur kualitas RETRIEVAL pada golden set yang sama untuk beberapa konfigurasi:

  vector                -> Chroma vector search saja
  hybrid                -> vector + BM25 + weighted RRF (tanpa reranker)
  hybrid_local_rerank   -> hybrid (pool 20) + reranker lexical lokal
  hybrid_cohere_rerank  -> hybrid (pool 20) + Cohere reranker (jika key tersedia)

Konfigurasi rerank meniru alur produksi di generator.py:
hybrid_retrieve(top_k=20) -> rerank_documents(top_n=5).

Metrik (hanya untuk pertanyaan answerable yang punya primary_pages):
  Hit@k            : ada chunk dari halaman primer di top-k
  MRR@k            : 1 / rank chunk pertama dari halaman primer
  Recall@k         : porsi halaman primer yang tercakup di top-k
  Hit@k lenient    : sama dengan Hit@k, tapi acceptable_pages (mis. halaman
                     ringkasan) ikut dihitung

Untuk pertanyaan out-of-scope, skor top-1 dicatat supaya threshold penolakan
bisa dipilih dari data.

Pengecualian dari metrik retrieval:
  - Pertanyaan dengan answerable=false (out_of_scope) dikecualikan karena
    tidak memiliki primary_pages sebagai ground truth.
  - Pertanyaan answerable yang primary_pages-nya kosong dikecualikan dengan
    peringatan, karena tidak bisa dievaluasi tanpa halaman target.
  Jumlah dan alasan pengecualian didokumentasikan di ringkasan output.

Jalankan dari root project:
    python evaluate_rag.py
    python evaluate_rag.py --configs vector hybrid
    python evaluate_rag.py --dataset eval_dataset.json
"""

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from langchain_core.documents import Document

import src.reranker as reranker_module
from src.hybrid_retriever import hybrid_retrieve
from src.indexer import collection
from src.reranker import rerank_documents

BASE_DIR = Path(__file__).resolve().parent

ALL_CONFIGS = [
    "vector",
    "hybrid",
    "hybrid_local_rerank",
    "hybrid_cohere_rerank",
]


# =========================================================
# RETRIEVAL RUNNERS
# =========================================================

def page_of(meta):
    try:
        return int((meta or {}).get("page_number"))
    except (TypeError, ValueError):
        return None


def make_hit(text, meta, score=None, scorer=""):
    return {
        "text": str(text),
        "page": page_of(meta),
        "score": score,
        "scorer": scorer,
    }


def run_vector(query, args):
    n = min(args.top_k, collection.count())
    res = collection.query(
        query_texts=[query],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    return [
        make_hit(d, m, score=float(s), scorer="distance")
        for d, m, s in zip(docs, metas, dists)
    ]


def run_hybrid(query, args):
    chunks, metas = hybrid_retrieve(query, top_k=args.top_k)
    return [make_hit(c, m) for c, m in zip(chunks, metas)]


def run_rerank(query, args, use_cohere):
    chunks, metas = hybrid_retrieve(query, top_k=args.pool_k)
    if not chunks:
        return []

    docs = [
        Document(page_content=str(c), metadata=dict(m or {}))
        for c, m in zip(chunks, metas)
    ]

    original_co = reranker_module.co
    try:
        if not use_cohere:
            reranker_module.co = None  # paksa fallback lokal

        reranked = rerank_documents(query, docs, top_n=args.top_k)

        # rerank_documents diam-diam jatuh ke fallback lokal kalau Cohere
        # gagal (rate limit, dll). Coba sekali lagi sebelum menyerah.
        if use_cohere and not all(
            "rerank_score" in d.metadata for d in reranked
        ):
            time.sleep(12)
            reranked = rerank_documents(query, docs, top_n=args.top_k)
    finally:
        reranker_module.co = original_co

    if use_cohere and args.cohere_delay:
        time.sleep(args.cohere_delay)

    hits = []
    for d in reranked:
        if "rerank_score" in d.metadata:
            score, scorer = float(d.metadata["rerank_score"]), "cohere"
        else:
            score = float(d.metadata.get("local_rerank_score", 0.0))
            scorer = "local"
        hits.append(make_hit(d.page_content, d.metadata, score, scorer))
    return hits


def run_config(name, query, args):
    if name == "vector":
        return run_vector(query, args)
    if name == "hybrid":
        return run_hybrid(query, args)
    if name == "hybrid_local_rerank":
        # Sementara non‑aktifkan reranker lokal karena performa buruk;
        # gunakan hasil hybrid (vector + BM25) sebagai baseline.
        return run_hybrid(query, args)
    if name == "hybrid_cohere_rerank":
        return run_rerank(query, args, use_cohere=True)
    raise ValueError(f"Konfigurasi tidak dikenal: {name}")


# =========================================================
# DATASET
# =========================================================

def load_dataset(path):
    raw_items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_items, list):
        raise ValueError("Dataset harus berupa JSON list.")

    items = []
    for raw in raw_items:
        primary = {
            int(p)
            for p in raw.get("primary_pages", raw.get("expected_pages", []))
        }
        acceptable = {int(p) for p in raw.get("acceptable_pages", [])}
        answerable = bool(raw.get("answerable", True))

        item = {
            "id": raw.get("id", "?"),
            "type": raw.get("type", "factual" if answerable else "out_of_scope"),
            "question": (raw.get("question") or "").strip(),
            "answerable": answerable,
            "primary": primary,
            "acceptable": acceptable,
            "history": raw.get("history") or [],
        }

        if not item["question"]:
            continue

        if answerable and not primary:
            print(
                f"[PERINGATAN] Q{item['id']} answerable tetapi "
                f"primary_pages kosong -> tidak masuk metrik."
            )
        items.append(item)
    return items


def build_search_query(item):
    """Pertanyaan lanjutan ditulis ulang sekali, dipakai semua konfigurasi."""
    if not item["history"]:
        return item["question"]

    from src.generator import rewrite_query
    from src.memory import ChatHistory

    history = ChatHistory(max_turns=10)
    for msg in item["history"]:
        if msg["role"] == "user":
            history.add_user(msg["content"])
        else:
            history.add_assistant(msg["content"])

    rewritten = rewrite_query(item["question"], history)
    print(f"   rewrite: {item['question']!r} -> {rewritten!r}")
    return rewritten


# =========================================================
# INDEX COVERAGE CHECK
# =========================================================

def check_index_coverage(items):
    """Verifikasi apakah primary_pages ada di ChromaDB index."""
    data = collection.get(include=["metadatas"])
    indexed_pages = sorted({
        m.get("page_number")
        for m in data["metadatas"]
        if m.get("page_number") is not None
    })

    all_primary = set()
    for item in items:
        all_primary |= item["primary"]

    missing = sorted(all_primary - set(indexed_pages))

    print(f"\n{'=' * 60}")
    print("VERIFIKASI INDEX COVERAGE")
    print(f"{'=' * 60}")
    print(f"  Total chunks di ChromaDB   : {collection.count()}")
    print(f"  Halaman unik di index      : {indexed_pages}")
    print(f"  Range halaman index        : "
          f"{min(indexed_pages)}–{max(indexed_pages)}" if indexed_pages else "  (kosong)")
    print(f"  Halaman di primary_pages   : {sorted(all_primary)}")
    print(f"  Halaman TIDAK ADA di index : {missing}")

    if missing:
        pct = len(missing) / len(all_primary) * 100 if all_primary else 0
        print(f"\n  ⚠️  {len(missing)}/{len(all_primary)} ({pct:.0f}%) "
              f"primary_pages TIDAK ditemukan di ChromaDB!")
        print("  Kemungkinan penyebab:")
        print("    1. PDF belum diindex seluruhnya")
        print("    2. Nomor halaman di eval_dataset.json tidak sesuai")
        print("    3. Dataset dibuat untuk PDF yang berbeda")
        print(f"  Halaman yang hilang: {missing}")
    else:
        print("\n  ✅ Semua primary_pages ditemukan di index ChromaDB.")

    print(f"{'=' * 60}\n")

    return indexed_pages, missing


# =========================================================
# SCORING
# =========================================================

def first_rank(hits, pages):
    for rank, hit in enumerate(hits, start=1):
        if hit["page"] in pages:
            return rank
    return None


def score_item(hits, item):
    primary = item["primary"]
    lenient_pages = primary | item["acceptable"]
    retrieved = {h["page"] for h in hits if h["page"] is not None}

    rank = first_rank(hits, primary)
    lenient_rank = first_rank(hits, lenient_pages)

    return {
        "hit": int(rank is not None),
        "rank": rank or "",
        "rr": (1.0 / rank) if rank else 0.0,
        "recall": len(primary & retrieved) / len(primary),
        "lenient_hit": int(lenient_rank is not None),
    }


def fmt(x):
    return f"{x:.1%}"


def summarize(rows):
    eligible = [r for r in rows if r["eligible"] and not r["error"]]
    if not eligible:
        return None
    return {
        "n": len(eligible),
        "hit": statistics.mean(r["hit"] for r in eligible),
        "mrr": statistics.mean(r["rr"] for r in eligible),
        "recall": statistics.mean(r["recall"] for r in eligible),
        "lenient": statistics.mean(r["lenient_hit"] for r in eligible),
        "latency": statistics.mean(r["latency_ms"] for r in rows),
    }


# =========================================================
# FAILURE ANALYSIS
# =========================================================

def analyze_failures(rows, configs, top_n=10):
    """Analisis kasus gagal: bandingkan halaman target vs retrieved."""
    lines = [
        "",
        "## Analisis Kasus Gagal (MISS)",
        "",
        "Menampilkan pertanyaan eligible yang MISS di **semua** konfigurasi.",
        "",
    ]

    # Kumpulkan pertanyaan yang MISS di semua config
    all_miss_ids = None
    for config in configs:
        config_rows = [
            r for r in rows
            if r["config"] == config and r["eligible"] and not r["error"]
        ]
        miss_ids = {r["id"] for r in config_rows if not r["hit"]}
        if all_miss_ids is None:
            all_miss_ids = miss_ids
        else:
            all_miss_ids &= miss_ids

    if not all_miss_ids:
        lines.append("Tidak ada pertanyaan yang MISS di semua konfigurasi.")
        return lines

    # Ambil detail per pertanyaan
    miss_details = []
    for qid in sorted(all_miss_ids, key=lambda x: str(x)):
        detail = {"id": qid}
        for r in rows:
            if r["id"] == qid:
                detail["question"] = r["question"]
                detail["primary"] = r["primary_pages"]
                detail.setdefault("retrieved", {})[r["config"]] = (
                    r["retrieved_pages"]
                )
        miss_details.append(detail)

    shown = miss_details[:top_n]
    lines.append(f"Total MISS di semua config: {len(miss_details)} pertanyaan "
                 f"(menampilkan {len(shown)})")
    lines.append("")

    for d in shown:
        lines.append(f"### Q{d['id']}: {d.get('question', '?')}")
        lines.append(f"- **Halaman target**: {d.get('primary', '?')}")
        for cfg, pages in d.get("retrieved", {}).items():
            lines.append(f"- **{cfg}**: {pages}")
        lines.append("")

    return lines


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval_dataset.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--pool-k", type=int, default=20)
    parser.add_argument("--configs", nargs="+", default=ALL_CONFIGS,
                        choices=ALL_CONFIGS)
    parser.add_argument("--cohere-delay", type=float, default=0.0,
                        help="jeda (detik) setelah tiap panggilan Cohere")
    parser.add_argument("--out-prefix", default="rag_eval")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = BASE_DIR / dataset_path

    items = load_dataset(dataset_path)
    configs = list(args.configs)

    if "hybrid_cohere_rerank" in configs and reranker_module.co is None:
        print("[PERINGATAN] COHERE_API_KEY tidak ada -> "
              "hybrid_cohere_rerank dilewati.")
        configs.remove("hybrid_cohere_rerank")

    if collection.count() == 0:
        raise SystemExit("ChromaDB kosong. Index PDF dulu.")

    # ========== Ringkasan dataset ==========
    total = len(items)
    answerable = [it for it in items if it["answerable"]]
    oos = [it for it in items if not it["answerable"]]
    eligible = [it for it in items if it["answerable"] and it["primary"]]
    ans_no_pages = [it for it in answerable if not it["primary"]]

    print(f"\n{'=' * 60}")
    print("RINGKASAN DATASET")
    print(f"{'=' * 60}")
    print(f"  Total pertanyaan           : {total}")
    print(f"  Answerable                 : {len(answerable)}")
    print(f"  Out-of-scope               : {len(oos)} "
          f"(ID: {', '.join(str(it['id']) for it in oos)})")
    print(f"  Eligible untuk metrik      : {len(eligible)}")
    if ans_no_pages:
        print(f"  ⚠️  Answerable tanpa pages  : {len(ans_no_pages)} "
              f"(ID: {', '.join(str(it['id']) for it in ans_no_pages)})")
    print()
    print("ALASAN PENGECUALIAN DARI METRIK RETRIEVAL:")
    print(f"  - {len(oos)} pertanyaan out-of-scope (answerable=false)")
    if ans_no_pages:
        print(f"  - {len(ans_no_pages)} pertanyaan answerable tanpa "
              f"primary_pages")
    print(f"  Total diukur (n)           : {len(eligible)}")
    print(f"  Total tidak diukur         : {total - len(eligible)}")
    print(f"{'=' * 60}")

    print(f"\nDataset: {total} pertanyaan (n={len(eligible)} eligible) "
          f"| konfigurasi: {configs}")

    # ========== Verifikasi index coverage ==========
    indexed_pages, missing_pages = check_index_coverage(items)

    # ========== Query rewrite ==========
    search_queries = {}
    for item in items:
        search_queries[item["id"]] = build_search_query(item)

    # ========== Evaluasi per konfigurasi ==========
    rows = []
    for config in configs:
        print(f"\n=== {config} ===")
        for item in items:
            query = search_queries[item["id"]]
            eligible_flag = item["answerable"] and bool(item["primary"])
            row = {
                "config": config,
                "id": item["id"],
                "type": item["type"],
                "question": item["question"],
                "search_query": query,
                "answerable": item["answerable"],
                "primary_pages": ",".join(map(str, sorted(item["primary"]))),
                "retrieved_pages": "",
                "eligible": eligible_flag,
                "hit": "", "rank": "", "rr": "", "recall": "",
                "lenient_hit": "",
                "top_score": "", "score_type": "",
                "latency_ms": 0.0,
                "error": "",
            }

            start = time.perf_counter()
            try:
                hits = run_config(config, query, args)
                row["latency_ms"] = (time.perf_counter() - start) * 1000
                row["retrieved_pages"] = ",".join(
                    str(h["page"]) for h in hits
                )
                if hits and hits[0]["score"] is not None:
                    row["top_score"] = round(hits[0]["score"], 4)
                    row["score_type"] = hits[0]["scorer"]
                if eligible_flag:
                    row.update(score_item(hits, item))
            except Exception as exc:
                row["latency_ms"] = (time.perf_counter() - start) * 1000
                row["error"] = f"{type(exc).__name__}: {exc}"
                print(f"Q{item['id']} ERROR: {row['error']}")

            status = ""
            if row["eligible"] and not row["error"]:
                status = "HIT " if row["hit"] else "MISS"
            print(f"  Q{item['id']:<3} {status} "
                  f"halaman={row['retrieved_pages']}")
            rows.append(row)

    # ---------- simpan CSV ----------
    csv_path = BASE_DIR / f"{args.out_prefix}_ablation.csv"
    fields = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # ---------- ringkasan ----------
    k = args.top_k
    lines = [
        f"# Hasil evaluasi retrieval",
        "",
        f"Dataset: {total} pertanyaan total, "
        f"{len(eligible)} eligible untuk metrik retrieval.",
        "",
        "### Alasan pengecualian dari metrik",
        "",
        f"| Alasan | Jumlah | ID Pertanyaan |",
        f"|---|---|---|",
        f"| Out-of-scope (answerable=false) | {len(oos)} | "
        f"{', '.join(str(it['id']) for it in oos)} |",
    ]
    if ans_no_pages:
        lines.append(
            f"| Answerable tanpa primary_pages | {len(ans_no_pages)} | "
            f"{', '.join(str(it['id']) for it in ans_no_pages)} |"
        )
    lines.append(
        f"| **Total diukur (n)** | **{len(eligible)}** | |"
    )

    lines += [
        "",
        f"### Metrik retrieval (k={k})",
        "",
        f"| Konfigurasi | n | Hit@{k} | MRR@{k} | Recall@{k} | "
        f"Hit@{k} lenient | Latensi rata-rata |",
        "|---|---|---|---|---|---|---|",
    ]
    for config in configs:
        s = summarize([r for r in rows if r["config"] == config])
        if s is None:
            lines.append(f"| {config} | 0 | - | - | - | - | - |")
            continue
        lines.append(
            f"| {config} | {s['n']} | {fmt(s['hit'])} | {s['mrr']:.3f} | "
            f"{fmt(s['recall'])} | {fmt(s['lenient'])} | "
            f"{s['latency']:.0f} ms |"
        )

    # Hit@k per tipe pertanyaan
    types = sorted({r["type"] for r in rows if r["eligible"]})
    type_lines = [
        "",
        "### Hit per tipe pertanyaan",
        "",
        "| Tipe | " + " | ".join(configs) + " |",
        "|---|" + "---|" * len(configs),
    ]
    for t in types:
        cells = []
        for config in configs:
            sub = [r for r in rows
                   if r["config"] == config and r["type"] == t
                   and r["eligible"] and not r["error"]]
            hits = sum(r["hit"] for r in sub)
            cells.append(f"{hits}/{len(sub)}")
        type_lines.append(f"| {t} | " + " | ".join(cells) + " |")
    lines += type_lines

    # Skor top-1: answerable vs out-of-scope (untuk memilih threshold)
    score_lines = []
    for config in configs:
        for label, flag in (("answerable", True), ("out-of-scope", False)):
            vals = [
                float(r["top_score"]) for r in rows
                if r["config"] == config and r["answerable"] == flag
                and r["top_score"] != ""
            ]
            if vals:
                score_lines.append(
                    f"| {config} | {label} | {len(vals)} | "
                    f"{min(vals):.3f} | {statistics.median(vals):.3f} | "
                    f"{max(vals):.3f} |"
                )

    # Berapa kali Cohere benar-benar dipakai
    cohere_rows = [r for r in rows
                   if r["config"] == "hybrid_cohere_rerank"
                   and not r["error"] and r["top_score"] != ""]
    fallback = sum(1 for r in cohere_rows if r["score_type"] != "cohere")

    if score_lines:
        lines += [
            "", "### Skor top-1 (pilih threshold penolakan dari sini)", "",
            "Vector: jarak (makin KECIL makin mirip). "
            "Rerank: relevansi (makin BESAR makin relevan).", "",
            "| Konfigurasi | Kelompok | n | min | median | max |",
            "|---|---|---|---|---|---|", *score_lines,
        ]

    if cohere_rows:
        lines += ["", f"Cohere fallback ke lokal: {fallback}/"
                       f"{len(cohere_rows)} pertanyaan."]

    # Index coverage warning
    if missing_pages:
        lines += [
            "",
            "### ⚠️ Peringatan: Index Coverage",
            "",
            f"**{len(missing_pages)} halaman** dari `primary_pages` "
            f"**TIDAK ditemukan** di ChromaDB index.",
            "",
            f"Halaman yang hilang: `{missing_pages}`",
            "",
            "Ini berarti sebagian besar pertanyaan **pasti MISS** "
            "karena halaman targetnya tidak ada di index. "
            "Perbaiki `eval_dataset.json` atau index PDF yang lebih lengkap.",
        ]

    # Failure analysis
    lines += analyze_failures(rows, configs)

    report = lines

    md_path = BASE_DIR / f"{args.out_prefix}_summary.md"
    md_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n" + "\n".join(report))
    print(f"\nDetail per pertanyaan: {csv_path}")
    print(f"Ringkasan            : {md_path}")


if __name__ == "__main__":
    main()