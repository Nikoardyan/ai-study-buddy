"""
fill_pages.py

Merapikan eval_dataset.json secara interaktif:
  1. Menghapus entri placeholder (yang pertanyaannya diawali "ISI PERTANYAAN").
  2. Mencari halaman PDF yang menyebut kata kunci tiap pertanyaan, lalu
     menampilkan kandidat + potongan teks. Kamu cukup mengetik nomor halaman
     yang benar-benar MENJELASKAN topiknya.
  3. Mengecek pertanyaan near_miss: kalau topiknya ternyata ada di PDF,
     kamu diperingatkan supaya pertanyaannya diganti.

File asli di-backup ke eval_dataset.backup.json.

Jalankan dari root project:
    python fill_pages.py
"""

import json
import re
import shutil
from pathlib import Path

from src.indexer import collection

DATASET = Path(__file__).resolve().parent / "eval_dataset.json"
BACKUP = DATASET.with_name("eval_dataset.backup.json")

# Kata kunci untuk mencari halaman tiap pertanyaan (id -> kata kunci).
FILL_KEYWORDS = {
    7: ["affordance"],
    8: ["affordance", "signifier"],
    11: ["efficiency", "efisiensi"],
    12: ["signifier"],
}

# Topik near_miss yang SEHARUSNYA tidak ada di PDF (id -> kata kunci).
NEAR_MISS_KEYWORDS = {
    13: ["system usability scale", "sus"],
    14: ["dark pattern", "dark patterns"],
    15: ["heuristik", "heuristic", "heuristics"],
}


def load_corpus():
    data = collection.get(include=["documents", "metadatas"])
    return list(zip(data["documents"], data["metadatas"]))


def find_pages(corpus, keywords):
    """Kembalikan {(kata, halaman): {count, snippet}} diurutkan per kata."""
    found = {}
    for doc, meta in corpus:
        page = (meta or {}).get("page_number")
        for kw in keywords:
            pattern = re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
            matches = list(pattern.finditer(doc))
            if not matches:
                continue
            entry = found.setdefault(
                (kw, page), {"count": 0, "snippet": ""}
            )
            entry["count"] += len(matches)
            if not entry["snippet"]:
                start = max(0, matches[0].start() - 80)
                end = matches[0].end() + 170
                entry["snippet"] = " ".join(doc[start:end].split())
    return found


def show_candidates(found, keywords):
    for kw in keywords:
        rows = sorted(
            ((page, e) for (k, page), e in found.items() if k == kw),
            key=lambda item: -item[1]["count"],
        )
        print(f"\n  Kata '{kw}':")
        if not rows:
            print("    (tidak ditemukan di PDF)")
        for page, entry in rows[:6]:
            print(f"    hal {page} ({entry['count']}x): "
                  f"...{entry['snippet']}...")


def ask_pages():
    while True:
        raw = input("\n  Halaman yang benar (pisah koma, mis. 9,10; "
                    "Enter = lewati): ").strip()
        if not raw:
            return None
        try:
            pages = sorted({int(x) for x in raw.replace(" ", "").split(",")})
            return pages
        except ValueError:
            print("  Format salah. Contoh: 9 atau 9,10")


def main():
    if collection.count() == 0:
        raise SystemExit("ChromaDB kosong. Index PDF dulu.")

    items = json.loads(DATASET.read_text(encoding="utf-8"))
    shutil.copy(DATASET, BACKUP)
    print(f"Backup: {BACKUP.name}")

    # 1. Hapus placeholder
    before = len(items)
    items = [
        it for it in items
        if not str(it.get("question", "")).startswith("ISI PERTANYAAN")
    ]
    print(f"Placeholder dihapus: {before - len(items)}")

    corpus = load_corpus()

    # 2. Isi primary_pages
    for item in items:
        needs_pages = item.get("answerable", True) and not item.get(
            "primary_pages"
        )
        if not needs_pages:
            continue

        keywords = FILL_KEYWORDS.get(item["id"])
        print("\n" + "=" * 70)
        print(f"Q{item['id']} [{item.get('type')}]: {item['question']}")

        if not keywords:
            print("  Tidak ada kata kunci otomatis. Isi manual di JSON.")
            continue

        show_candidates(find_pages(corpus, keywords), keywords)
        pages = ask_pages()
        if pages:
            item["primary_pages"] = pages
            item.pop("_todo", None)
            print(f"  -> primary_pages = {pages}")
        else:
            print("  -> dilewati (masih kosong)")

    # 3. Cek near_miss
    print("\n" + "=" * 70)
    print("CEK PERTANYAAN NEAR_MISS (topiknya seharusnya TIDAK ada di PDF)")
    for item in items:
        keywords = NEAR_MISS_KEYWORDS.get(item["id"])
        if item.get("type") != "near_miss" or not keywords:
            continue
        found = find_pages(corpus, keywords)
        if found:
            pages = sorted({page for (_, page) in found})
            print(f"\nQ{item['id']}: {item['question']}")
            print(f"  PERINGATAN: topik ditemukan di halaman {pages}. "
                  f"Ganti pertanyaan ini dengan topik yang benar-benar "
                  f"tidak ada di PDF.")
            show_candidates(found, keywords)
        else:
            print(f"Q{item['id']}: OK, topik tidak ada di PDF.")
            item.pop("_todo", None)

    DATASET.write_text(
        json.dumps(items, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    remaining = [
        it["id"] for it in items
        if it.get("answerable", True) and not it.get("primary_pages")
    ]
    print(f"\nTersimpan: {DATASET.name} ({len(items)} pertanyaan)")
    if remaining:
        print(f"Masih tanpa primary_pages: {remaining}")
    else:
        print("Semua pertanyaan answerable sudah punya primary_pages.")


if __name__ == "__main__":
    main()