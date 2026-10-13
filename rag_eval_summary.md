# Hasil evaluasi retrieval

Dataset: 30 pertanyaan total, 23 eligible untuk metrik retrieval.

### Alasan pengecualian dari metrik

| Alasan | Jumlah | ID Pertanyaan |
|---|---|---|
| Out-of-scope (answerable=false) | 7 | 3, 6, 9, 14, 19, 23, 28 |
| **Total diukur (n)** | **23** | |

### Metrik retrieval (k=5)

| Konfigurasi | n | Hit@5 | MRR@5 | Recall@5 | Hit@5 lenient | Latensi rata-rata |
|---|---|---|---|---|---|---|
| hybrid | 23 | 91.3% | 0.636 | 89.9% | 100.0% | 121 ms |

### Hit per tipe pertanyaan

| Tipe | hybrid |
|---|---|
| factual | 15/17 |
| followup | 4/4 |
| multi_chunk | 2/2 |

## Analisis Kasus Gagal (MISS)

Menampilkan pertanyaan eligible yang MISS di **semua** konfigurasi.

Total MISS di semua config: 2 pertanyaan (menampilkan 2)

### Q2: Sebutkan komponen utama IMK.
- **Halaman target**: 4
- **hybrid**: 19,12,21,18,21

### Q24: Apa tujuan IMK?
- **Halaman target**: 3
- **hybrid**: 19,21,21,20,19
