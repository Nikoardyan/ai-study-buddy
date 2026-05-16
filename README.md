# 🎓 AI Study Buddy

> Teman belajar berbasis AI yang menjawab pertanyaan dari materi yang kamu upload — didukung RAG pipeline dan DeepSeek LLM.

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57-red?style=flat-square&logo=streamlit)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-green?style=flat-square)
![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-purple?style=flat-square)

---

## Tentang Proyek

AI Study Buddy adalah aplikasi chatbot edukatif yang menggunakan **Retrieval-Augmented Generation (RAG)** untuk menjawab pertanyaan siswa berdasarkan materi belajar yang mereka upload sendiri. Bukan hanya menjawab — AI ini dirancang seperti kakak tutor: memberikan penjelasan bertahap, analogi sehari-hari, tips trik, dan soal refleksi.

**Masalah yang diselesaikan:** Siswa sering kesulitan memahami materi pelajaran tanpa bimbingan. AI Study Buddy hadir sebagai tutor 24/7 yang benar-benar memahami isi buku/modul mereka.

---

## Fitur Utama

| Fitur | Deskripsi |
|---|---|
| **Upload Multi-format** | Mendukung PDF, DOCX, TXT, dan Markdown |
| **Auto-detect Subject** | Otomatis mendeteksi mata pelajaran dari nama file |
| **RAG Pipeline** | Retrieval berbasis vektor untuk jawaban akurat dan relevan |
| **Streaming Response** | Jawaban muncul real-time kata per kata |
| **Citation Otomatis** | Setiap jawaban dilengkapi sumber `[1]`, `[2]`, dst. |
| **Conversation Memory** | Menyimpan konteks percakapan hingga 10 turn terakhir |
| **Manajemen Dokumen** | Lihat dan hapus materi langsung dari sidebar |
| **Dark UI Modern** | Tampilan glassmorphism dengan gradient ungu |

---

## Arsitektur

```
User Question
     │
     ▼
┌─────────────────────────────────────────┐
│              Streamlit UI               │
│  (app.py — upload, chat, sidebar stats) │
└──────────────────┬──────────────────────┘
                   │
       ┌───────────┴───────────┐
       ▼                       ▼
┌─────────────┐        ┌──────────────────┐
│   Indexing  │        │    Retrieval +   │
│   Pipeline  │        │    Generation    │
│             │        │                  │
│ loader.py   │        │  generator.py    │
│ chunker.py  │        │  memory.py       │
│ indexer.py  │        │                  │
└──────┬──────┘        └───────┬──────────┘
       │                       │
       ▼                       ▼
┌─────────────────────────────────────────┐
│           ChromaDB (Vector Store)       │
│         ./chroma_db/chroma.sqlite3      │
└─────────────────────────────────────────┘
                   │
                   ▼
         ┌─────────────────┐
         │   DeepSeek LLM  │
         │  (via OpenAI    │
         │  compatible API)│
         └─────────────────┘
```

### Komponen Utama

```
src/
├── config.py      # Konfigurasi API, model, chunking, retrieval
├── loader.py      # Parsing dokumen (PDF, DOCX, TXT, MD)
├── chunker.py     # Smart chunking dengan overlap
├── indexer.py     # Indexing ke ChromaDB + auto-detect subject
├── generator.py   # RAG answer generation dengan streaming
└── memory.py      # Manajemen conversation history
```

---

## Tech Stack

- **Frontend / UI:** [Streamlit](https://streamlit.io/)
- **Vector Database:** [ChromaDB](https://www.trychroma.com/)
- **LLM:** [DeepSeek](https://www.deepseek.com/) via OpenAI-compatible API
- **Document Parsing:** `pdfplumber`, `pypdf`, `python-docx`
- **Text Splitting:** `langchain-text-splitters`
- **HTTP Client:** `openai` SDK

---

## Persyaratan

- Python 3.11+
- DeepSeek API Key (atau API key lain yang kompatibel dengan OpenAI format)

---

## Instalasi

### 1. Clone repository

```bash
git clone https://github.com/username/ai-study-buddy.git
cd ai-study-buddy
```

### 2. Buat virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Konfigurasi environment variables

Buat file `.env` di root proyek:

```env
OPENAI_API_KEY=your_deepseek_api_key_here
OPENAI_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

> **Catatan:** Variabel menggunakan prefix `OPENAI_` karena DeepSeek menyediakan API yang kompatibel dengan format OpenAI.

---

## Menjalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka di `http://localhost:8501`.

---

## Cara Penggunaan

1. **Upload Materi** — Klik area upload di sidebar, pilih file PDF/DOCX/TXT/MD, lalu klik **Upload & Index**. AI akan otomatis memecah dokumen menjadi chunks dan menyimpannya ke vector database.

2. **Naming Convention (Opsional)** — Penamaan file dengan prefix subject membantu auto-detect:
   ```
   01_matematika_kalkulus_diferensial.pdf
   02_fisika_hukum_newton.pdf
   03_python_dasar_pemrograman.pdf
   ```

3. **Bertanya** — Ketik pertanyaan di kolom chat. AI akan mencari chunk paling relevan dan menjawab dengan gaya tutor + citation.

4. **Filter Subject** — Sebutkan mata pelajaran dalam pertanyaan untuk hasil yang lebih fokus:
   > *"Jelaskan hukum Newton pertama dari fisika"*
   > *"Berikan contoh soal turunan matematika"*

---

## Subject yang Didukung

| Subject | Keywords Pendeteksi |
|---|---|
| Matematika | `matematika`, `math`, `kalkulus`, `aljabar`, `geometri`, `statistik` |
| Fisika | `fisika`, `physics`, `newton`, `mekanika`, `termodinamika` |
| Kimia | `kimia`, `chemistry`, `reaksi`, `molekul` |
| Biologi | `biologi`, `biology`, `sel`, `genetika` |
| Python | `python`, `programming`, `coding`, `algoritma` |
| Sejarah | `sejarah`, `history` |
| Ekonomi | `ekonomi`, `economics` |
| Bahasa | `bahasa`, `grammar`, `sastra` |

---

## Konfigurasi Lanjutan

Edit `src/config.py` untuk menyesuaikan perilaku sistem:

```python
CHUNK_SIZE    = 800    # Ukuran chunk teks (karakter)
CHUNK_OVERLAP = 100    # Overlap antar chunk
TOP_K         = 4      # Jumlah chunk yang di-retrieve per pertanyaan
TEMPERATURE   = 0.4    # Kreativitas LLM (0 = deterministik, 1 = kreatif)
MAX_TOKENS    = 700    # Panjang maksimum jawaban
```

---

## Struktur Proyek

```
ai-study-buddy/
├── app.py              # Aplikasi Streamlit utama
├── api.py              # FastAPI server (opsional)
├── main.py             # CLI entry point
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (tidak di-commit)
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── loader.py
│   ├── chunker.py
│   ├── indexer.py
│   ├── generator.py
│   └── memory.py
├── docs/               # Contoh materi belajar
│   ├── 01_matematika_kalkulus_diferensial.pdf
│   └── 02_fisika_hukum_newton.pdf
├── static/
│   └── index.html
└── chroma_db/          # Vector database (auto-generated)
    └── chroma.sqlite3
```

---

## Lisensi

MIT License — bebas digunakan untuk keperluan edukasi dan pengembangan.
