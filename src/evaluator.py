"""
evaluate_answers.py

Evaluasi END-TO-END (jawaban akhir), bukan hanya retrieval.

Tahap 1 - jalankan pipeline apa adanya (tanpa cache) dan simpan jawaban:
    python evaluate_answers.py run --dataset eval_dataset.json
    -> answers_for_grading.csv

Tahap 2 - buka CSV itu di Excel/Sheets dan isi kolom `grade` secara manual:
    pertanyaan answerable   : benar | sebagian | salah | mengarang
    pertanyaan out-of-scope : tolak_benar | menjawab_salah
    (kolom refused_auto hanya tebakan otomatis dari frasa; nilai manual
     yang jadi acuan)

Tahap 3 - hitung hasil:
    python evaluate_answers.py score
"""

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUT_PATH = BASE_DIR / "answers_for_grading.csv"


def cmd_run(args):
    import src.generator as gen
    from src.memory import ChatHistory

    # Matikan cache supaya setiap jawaban benar-benar dihasilkan ulang.
    gen.check_cache = lambda *a, **k: None
    gen.save_to_cache = lambda *a, **k: None

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = BASE_DIR / dataset_path
    items = json.loads(dataset_path.read_text(encoding="utf-8"))

    rows = []
    for item in items:
        question = (item.get("question") or "").strip()
        if not question:
            continue

        history = ChatHistory(max_turns=10)
        for msg in item.get("history") or []:
            if msg["role"] == "user":
                history.add_user(msg["content"])
            else:
                history.add_assistant(msg["content"])

        answerable = bool(item.get("answerable", True))
        parts, first_token, error = [], None, ""
        start = time.perf_counter()

        try:
            for token in gen.generate_answer_streaming(
                question, chat_history=history
            ):
                if first_token is None:
                    first_token = time.perf_counter() - start
                parts.append(token)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        total = time.perf_counter() - start
        answer = "".join(parts)

        rows.append({
            "id": item.get("id", "?"),
            "type": item.get("type", "factual" if answerable else "out_of_scope"),
            "question": question,
            "answerable": answerable,
            "answer": answer,
            "refused_auto": gen._answer_says_not_found(answer),
            "has_page_citation": ("**Sumber materi:**" in answer
                                  and "halaman" in answer),
            "first_token_s": round(first_token, 2) if first_token else "",
            "total_s": round(total, 2),
            "error": error,
            "grade": "",
            "notes": "",
        })
        print(f"Q{rows[-1]['id']:<3} {total:5.1f}s  "
              f"refused_auto={rows[-1]['refused_auto']}  {question[:60]}")
        time.sleep(args.delay)

    with OUT_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    totals = [r["total_s"] for r in rows if not r["error"]]
    firsts = [r["first_token_s"] for r in rows if r["first_token_s"] != ""]
    if totals:
        print(f"\nLatensi total rata-rata: {statistics.mean(totals):.2f}s "
              f"| median {statistics.median(totals):.2f}s")
    if firsts:
        print(f"Waktu ke token pertama : {statistics.mean(firsts):.2f}s")
    print(f"\nIsi kolom `grade` di: {OUT_PATH}")


def cmd_score(args):
    path = Path(args.file) if args.file else OUT_PATH
    with path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    graded = [r for r in rows if r["grade"].strip()]
    if not graded:
        raise SystemExit("Kolom `grade` masih kosong.")

    answerable = [r for r in graded if r["answerable"] == "True"]
    oos = [r for r in graded if r["answerable"] != "True"]

    print(f"Terisi: {len(graded)}/{len(rows)} baris\n")

    if answerable:
        counts = {g: sum(r["grade"] == g for r in answerable)
                  for g in ("benar", "sebagian", "salah", "mengarang")}
        n = len(answerable)
        faithful = counts["benar"] + counts["sebagian"]
        print(f"PERTANYAAN DALAM DOKUMEN (n={n})")
        for g, c in counts.items():
            print(f"  {g:<10}: {c}")
        print(f"  Setia pada sumber (benar+sebagian): "
              f"{faithful}/{n} = {faithful / n:.1%}")
        print(f"  Benar penuh                       : "
              f"{counts['benar']}/{n} = {counts['benar'] / n:.1%}")

    if oos:
        ok = sum(r["grade"] == "tolak_benar" for r in oos)
        print(f"\nPERTANYAAN DI LUAR DOKUMEN (n={len(oos)})")
        print(f"  Ditolak dengan benar: {ok}/{len(oos)} = {ok / len(oos):.1%}")

        by_type = {}
        for r in oos:
            by_type.setdefault(r["type"], []).append(r)
        for t, group in sorted(by_type.items()):
            good = sum(r["grade"] == "tolak_benar" for r in group)
            print(f"    {t:<14}: {good}/{len(group)}")

        # Seberapa akurat deteksi otomatis frasa penolakan?
        agree = sum(
            (r["refused_auto"] == "True") == (r["grade"] == "tolak_benar")
            for r in oos
        )
        print(f"  Kesesuaian deteksi otomatis vs penilaian manual: "
              f"{agree}/{len(oos)}")

    totals = [float(r["total_s"]) for r in rows if r["total_s"]]
    if totals:
        print(f"\nLatensi total rata-rata: {statistics.mean(totals):.2f}s")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run")
    run.add_argument("--dataset", default="eval_dataset.json")
    run.add_argument("--delay", type=float, default=1.0,
                     help="jeda antar pertanyaan (hindari rate limit Groq)")
    run.set_defaults(func=cmd_run)

    score = sub.add_parser("score")
    score.add_argument("--file", default=None)
    score.set_defaults(func=cmd_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()