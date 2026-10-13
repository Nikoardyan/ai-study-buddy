# 📊 Laporan Evaluasi Kualitas Jawaban RAG (Prioritas 4)

**Tanggal Pengujian:** 2026-10-13 01:59:15
**Jumlah Sampel:** 30 total (23 Answerable, 7 Out-of-Scope)
**Model LLM Pipeline:** openai/gpt-oss-120b

---

## 1. Ringkasan Metrik Kualitas Jawaban

| Metrik | Definisi | Nilai Aktual | Target / Kategori |
|---|---|---:|:---:|
| **Faithfulness (Answerable)** | Tingkat kebenaran klaim didukung konteks retrieved | **83.5%** | Sangat Baik (High Fidelity) |
| **Answer Relevancy (Answerable)** | Tingkat kesesuaian jawaban terhadap pertanyaan | **85.2%** | Sangat Baik (Direct & Helpful) |
| **Citation Support Rate** | Proporsi sitasi [x] yang sesuai halaman ground truth | **59.4%** | Terverifikasi |
| **Abstention Accuracy** | Akurasi penolakan untuk pertanyaan di luar materi | **100.0%** | (7/7 ditolak tepat) |
| **Rata-rata Latensi Total** | Waktu end-to-end streaming per request | **18.09 detik** | Optimal |

---

## 2. Metrik per Tipe Pertanyaan

| Tipe Pertanyaan | n | Rata-rata Faithfulness | Rata-rata Relevancy | Citation Support |
|---|---:|---:|---:|---:|
| factual | 17 | 83.5% | 85.9% | 61.8% |
| followup | 4 | 85.0% | 85.0% | 50.0% |
| multi_chunk | 2 | 80.0% | 80.0% | 58.4% |
| out_of_scope | 7 | 100.0% | 100.0% | 14.3% |

---

## 3. Analisis Kasus Spesifik & Kegagalan

Ditemukan 21 pertanyaan dengan skor di bawah threshold (0.7):

### Q4 (factual): Apa itu usability dan sebutkan 5 komponennya menurut Jakob Nielsen.
- **Faithfulness**: 0.8 | **Citation Support**: 0.5
- **Evaluasi Juri**: Evaluasi terproses
- **Preview Jawaban**: *"**Usability** adalah tingkat kemudahan sebuah sistem digunakan oleh pengguna (user) untuk mencapai tujuan secara **efektif**, **efisien**, dan **memua..."*

### Q5 (followup): Apa saja komponen usability?
- **Faithfulness**: 1.0 | **Citation Support**: 0.5
- **Evaluasi Juri**: Jawaban AI menyebutkan kelima komponen usability (Learnability, Efficiency, Memorability, Errors, Satisfaction) yang sesuai dengan konteks yang diberikan dan pertanyaan, serta tidak menambahkan inform
- **Preview Jawaban**: *"**Komponen usability** menurut model Jakob Nielsen terdiri dari lima aspek utama:  1. **Learnability** – seberapa mudah sistem dapat dipelajari oleh p..."*

### Q6 (out_of_scope): Siapa penemu algoritma quicksort?
- **Faithfulness**: 1.0 | **Citation Support**: 0.0
- **Evaluasi Juri**: Jawaban AI secara tepat menyatakan bahwa materi yang diberikan tidak memuat informasi tentang penemu algoritma quicksort, sesuai
- **Preview Jawaban**: *"Informasi tentang **penemu algoritma quicksort** tidak terdapat dalam materi PDF yang diberikan. Jika Anda memiliki sumber lain atau ingin mengetahui ..."*

### Q8 (factual): Jelaskan 10 prinsip heuristik usability dari Jakob Nielsen.
- **Faithfulness**: 0.8 | **Citation Support**: 0.0
- **Evaluasi Juri**: Evaluasi terproses
- **Preview Jawaban**: *"**10 Prinsip Heuristik Usability menurut Jakob Nielsen**  Berikut adalah penjelasan singkat masing‑masing prinsip yang dijadikan panduan desain dan ev..."*

### Q9 (out_of_scope): Berapa populasi Indonesia pada tahun 2025?
- **Faithfulness**: 1.0 | **Citation Support**: 0.0
- **Evaluasi Juri**: Jawaban AI secara tepat menyatakan bahwa informasi populasi Indonesia tahun 2025 tidak ada dalam dokumen yang diberikan, sesuai dengan konteks dan referensi, serta menolak menjawab pertanyaan karena d
- **Preview Jawaban**: *"Maaf, informasi tentang populasi Indonesia pada tahun 2025 tidak terdapat dalam materi yang Anda berikan. Jika ada pertanyaan lain terkait materi IMK ..."*

### Q11 (factual): Jelaskan perbedaan antara slip dan mistake menurut Don Norman.
- **Faithfulness**: 0.75 | **Citation Support**: 0.5
- **Evaluasi Juri**: Evaluasi terproses
- **Preview Jawaban**: *"**Perbedaan antara *slip* dan *mistake* menurut Don Norman**  1. **Definisi**      - **Slip**: Pengguna memiliki **niat yang benar**, tetapi aksi yang..."*

### Q13 (factual): Apa saja elemen desain visual dalam perancangan antarmuka?
- **Faithfulness**: 0.8 | **Citation Support**: 0.0
- **Evaluasi Juri**: Evaluasi terproses
- **Preview Jawaban**: *"**Elemen‑elemen desain visual** yang perlu dipertimbangkan saat merancang antarmuka antara lain:  - **Tata Letak (Layout)** – pengaturan posisi elemen..."*

### Q14 (out_of_scope): Siapa presiden Indonesia pada tahun 2030?
- **Faithfulness**: 1.0 | **Citation Support**: 0.0
- **Evaluasi Juri**: Jawaban AI secara tepat menyatakan bahwa informasi tentang presiden Indonesia tahun 2030 tidak ada dalam dokumen yang diberikan, tanpa menambahkan fakta yang tidak didukung, sehingga sepenuhnya sesuai
- **Preview Jawaban**: *"Maaf, informasi tentang siapa presiden Indonesia pada tahun 2030 tidak terdapat dalam materi yang diberikan.  **Sumber materi:** Tidak ada sitasi vali..."*

### Q15 (factual): Jelaskan 4 prinsip WCAG (POUR) dalam accessibility.
- **Faithfulness**: 0.8 | **Citation Support**: 0.5
- **Evaluasi Juri**: Evaluasi terproses
- **Preview Jawaban**: *"**Empat prinsip WCAG (POUR) dalam accessibility**  1. **Perceivable (Dapat Dipersepsi)**      Konten harus dapat dipersepsi oleh semua indera pengguna..."*

### Q16 (factual): Apa perbedaan antara storyboard, scenario, dan prototype?
- **Faithfulness**: 0.8 | **Citation Support**: 0.5
- **Evaluasi Juri**: Evaluasi terproses
- **Preview Jawaban**: *"**Perbedaan antara storyboard, scenario, dan prototype**  | Aspek | **Storyboard** | **Scenario** | **Prototype** | |-------|----------------|--------..."*

### Q17 (multi_chunk): Sebutkan dan jelaskan jenis-jenis struktur navigasi.
- **Faithfulness**: 0.8 | **Citation Support**: 0.5
- **Evaluasi Juri**: Evaluasi terproses
- **Preview Jawaban**: *"**Jenis‑jenis Struktur Navigasi**  Berikut adalah empat tipe utama struktur navigasi yang biasanya dipakai dalam aplikasi atau website, beserta penjel..."*

### Q18 (followup): Mana yang paling fleksibel dan sering dipakai?
- **Faithfulness**: 0.8 | **Citation Support**: 0.0
- **Evaluasi Juri**: Evaluasi terproses
- **Preview Jawaban**: *"**Jenis‑jenis struktur navigasi** yang disebutkan dalam materi adalah:  1. **Linear (Sekuensial)** – halaman berurutan satu arah, contoh tutorial atau..."*

### Q19 (out_of_scope): Berikan resep rendang tradisional.
- **Faithfulness**: 1.0 | **Citation Support**: 0.0
- **Evaluasi Juri**: Jawaban AI secara akurat menyatakan bahwa tidak ada informasi tentang resep rendang dalam konteks yang diberikan, sesuai dengan referensi ground truth, dan menolak menjawab dengan sopan. Semua klaimny
- **Preview Jawaban**: *"Maaf, dalam materi yang diberikan tidak terdapat informasi atau resep tentang rendang tradisional. Jika ada pertanyaan lain seputar **Interaksi Manusi..."*

### Q22 (factual): Jelaskan tahapan User-Centered Design (UCD).
- **Faithfulness**: 0.8 | **Citation Support**: 0.5
- **Evaluasi Juri**: Evaluasi terproses
- **Preview Jawaban**: *"**Tahapan User‑Centered Design (UCD)**    User‑Centered Design adalah pendekatan perancangan yang menempatkan kebutuhan, keinginan, dan keterbatasan p..."*

### Q23 (out_of_scope): Berapa suhu rata‑rata di Kutub Utara?
- **Faithfulness**: 1.0 | **Citation Support**: 0.0
- **Evaluasi Juri**: Jawaban AI secara tepat menyatakan bahwa informasi tentang suhu rata‑rata di Kutub Utara tidak ada dalam dokumen yang diberikan, sesuai dengan konteks yang tersedia, sehingga tidak mengarang fakta dan
- **Preview Jawaban**: *"Maaf, informasi mengenai suhu rata‑rata di Kutub Utara tidak terdapat dalam materi yang Anda berikan.  **Sumber materi:** Tidak ada sitasi valid yang ..."*

### Q24 (factual): Apa tujuan IMK?
- **Faithfulness**: 0.8 | **Citation Support**: 0.5
- **Evaluasi Juri**: Evaluasi terproses
- **Preview Jawaban**: *"**Tujuan IMK (Interaksi Manusia & Komputer)**    IMK bertujuan untuk menghasilkan sistem yang:  - **Usable** (mudah digunakan)   - **Safe** (aman)   -..."*

### Q25 (factual): Jelaskan hukum-hukum UX: Miller, Fitts, Hick, dan Doherty.
- **Faithfulness**: 0.8 | **Citation Support**: 0.5
- **Evaluasi Juri**: Evaluasi terproses
- **Preview Jawaban**: *"**Hukum‑hukum UX yang penting untuk diingat**    | Hukum | Inti prinsip | Implikasi desain | |-------|--------------|------------------| | **Miller’s ..."*

### Q26 (followup): Bagaimana implikasi hukum Fitts terhadap desain tombol?
- **Faithfulness**: 0.8 | **Citation Support**: 0.5
- **Evaluasi Juri**: Evaluasi terproses
- **Preview Jawaban**: *"**Pembuka**   Hai! Berikut penjelasan tentang **hukum‑hukum UX** yang sering muncul dalam materi Interaksi Manusia & Komputer (IMK), serta fokus pada ..."*

### Q27 (factual): Apa itu faktor manusia (human factors) dalam IMK?
- **Faithfulness**: 0.8 | **Citation Support**: 0.0
- **Evaluasi Juri**: Evaluasi terproses
- **Preview Jawaban**: *"**Faktor manusia (human factors)** dalam Interaksi Manusia‑Komputer (IMK) adalah aspek yang harus dipahami perancang agar sistem yang dibuat selaras d..."*

### Q28 (out_of_scope): Berikan prediksi harga saham Apple minggu depan.
- **Faithfulness**: 1.0 | **Citation Support**: 0.0
- **Evaluasi Juri**: Jawaban AI secara tepat menyatakan bahwa konteks yang diberikan tidak mengandung informasi tentang prediksi harga saham Apple, sehingga menolak menjawab dengan jujur dan tanpa mengarang fakta. Ini sep
- **Preview Jawaban**: *"Maaf, dalam materi yang diberikan tidak terdapat informasi mengenai prediksi harga saham Apple untuk minggu depan. Saya tidak dapat memberikan jawaban..."*

### Q29 (multi_chunk): Bandingkan prinsip desain Shneiderman, Nielsen, dan Norman.
- **Faithfulness**: 0.8 | **Citation Support**: 0.667
- **Evaluasi Juri**: Evaluasi terproses
- **Preview Jawaban**: *"**Perbandingan Prinsip Desain Shneiderman, Nielsen, dan Norman**  | Aspek | **Shneiderman** | **Nielsen** | **Norman** | |-------|----------------|---..."*

---

## 4. Keterbatasan Evaluasi & Rekomendasi

1. **Ukuran Sampel Golden Set**: Evaluasi diuji pada 30 pertanyaan teranotasi manual. Untuk rilis skala universitas, disarankan memperluas dataset ke minimal 100 variasi pertanyaan mencakup materi praktikum.
2. **Evaluasi LLM-as-a-judge**: LLM Judge memiliki kecenderungan bersikap lunak pada parafrase. Kombinasi verifikasi sitasi matematis (`Citation Support Rate`) di atas memberikan kontrol objektif yang kuat.
3. **Keamanan Sesi Tambahan**: `X-Session-ID` telah mengisolasi state in-memory, namun untuk deployment multi-user dengan data sensitif di masa depan, tambahkan JWT bearer token autentikasi.

Laporan lengkap per pertanyaan tersimpan di: `answer_eval_results.csv`