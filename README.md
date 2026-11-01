# Telco Customer Churn Prediction (PA-1A)

Model klasifikasi biner untuk memprediksi pelanggan telekomunikasi yang berpotensi berhenti berlangganan (*churn*), lalu mengubah skor risikonya menjadi daftar prioritas untuk program retensi. Proyek mencakup business understanding, data quality check, EDA, feature engineering, modeling, tuning, optimasi threshold berbasis biaya, interpretasi model, dan estimasi dampak bisnis.

> **Ringkasan hasil:** model final **XGBoost** dengan threshold **0,41** mencapai ROC-AUC **0,843** dan PR-AUC **0,671** pada test set. Pada asumsi kampanye yang transparan, estimasi *net benefit* adalah **+9.916**, dibanding **−4.246** bila pelanggan dihubungi secara acak.

---

## Daftar Isi

1. [Latar belakang bisnis](#latar-belakang-bisnis)
2. [Dataset](#dataset)
3. [Alur kerja](#alur-kerja)
4. [Temuan utama EDA](#temuan-utama-eda)
5. [Hasil modeling](#hasil-modeling)
6. [Interpretasi model](#interpretasi-model)
7. [Dampak bisnis dan rekomendasi](#dampak-bisnis-dan-rekomendasi)
8. [Keterbatasan](#keterbatasan)
9. [Struktur repositori](#struktur-repositori)
10. [Cara menjalankan](#cara-menjalankan)
11. [Reproducibility](#reproducibility)
12. [Penggunaan AI](#penggunaan-ai)
13. [Penulis](#penulis)

---

## Latar belakang bisnis

Sekitar **26,5%** pelanggan pada dataset ini churn (1.869 dari 7.043). Tim retensi memiliki anggaran dan kapasitas terbatas sehingga tidak mungkin menghubungi semua pelanggan. Model ini membantu menjawab satu keputusan: **pelanggan mana yang sebaiknya dihubungi pada tiap siklus kampanye retensi.**

| Aspek | Definisi |
|---|---|
| Target | `Churn`: `Yes` = 1 (berhenti), `No` = 0 (bertahan) |
| Jenis masalah | Klasifikasi biner dengan imbalance moderat (73,5% : 26,5%) |
| Metrik model | ROC-AUC, PR-AUC, Precision, Recall, F1-Score, Confusion Matrix |
| Metrik bisnis | Net benefit kampanye, expected saved revenue, lift top-k%, churn rate |

*Accuracy* sengaja tidak dijadikan metrik utama: menebak "tidak churn" untuk semua pelanggan sudah 73,5% benar tanpa menangkap satu pun pelanggan churn.

## Dataset

- **Sumber:** Telco Customer Churn (IBM, via Kaggle), file `data/1A_telco_customer_churn.csv`
- **Ukuran:** 7.043 baris, 21 kolom
- **Isi:** profil demografi, layanan yang digunakan, jenis kontrak, metode pembayaran, tagihan, dan status churn

**Masalah kualitas data yang ditemukan dan penanganannya**

| Temuan | Penanganan |
|---|---|
| `TotalCharges` bertipe teks; 11 baris berisi string kosong (semuanya `tenure = 0`, pelanggan baru) | Dikonversi ke angka; kosong diisi 0 karena pelanggan belum pernah ditagih |
| Label "No phone service" / "No internet service" redundan (682 dan 1.526 baris) | Disederhanakan menjadi "No" |
| Duplikat dan outlier (IQR) | Tidak ada; tidak ada baris yang dihapus |
| Imbalance kelas | Stratified split, threshold tuning |

## Alur kerja

1. **Business understanding & data diagnosis:** problem statement, metrik, tabel data quality, 9 visualisasi EDA dengan insight tertulis.
2. **Data preparation:** cleaning, 5 fitur baru, seleksi fitur, `Pipeline` + `ColumnTransformer`, split stratified 70/15/15.
3. **Modeling:** Dummy (baseline), Logistic Regression, Random Forest, XGBoost; perbandingan penanganan imbalance; tuning dengan `RandomizedSearchCV`; optimasi threshold berbasis kapasitas dan biaya; evaluasi test sekali pakai.
4. **Evaluasi & interpretasi:** pemilihan model final, error analysis, permutation importance, SHAP, koefisien Logistic Regression, perhitungan dampak bisnis, rekomendasi.

**Fitur baru:** `ServiceCount`, `TenureGroup`, `IsLongTermContract`, `IsAutoPay`, `PriceChangeRatio`. Semuanya dihitung per baris tanpa memakai target.

**Seleksi fitur** (hanya pada data train): 25 fitur menjadi 22 (dibuang: `gender`, `PhoneService`, `AverageMonthlySpend`), menjadi 32 kolom setelah encoding.

**Pencegahan data leakage**
- Split dilakukan sebelum seleksi fitur dan scaling.
- `StandardScaler` dan `OneHotEncoder` hanya di-fit pada train, di dalam `Pipeline`.
- Threshold ditentukan di validation set; test set dipakai satu kali.
- 11 pemeriksaan otomatis (`assert`) di notebook, misalnya mean scaler sama dengan mean train dan berbeda dari mean seluruh data.

| Split | n | Churn rate |
|---|---|---|
| Train | 4.930 | 26,5% |
| Validation | 1.056 | 26,5% |
| Test | 1.057 | 26,6% |

## Temuan utama EDA

| Faktor | Churn rate |
|---|---|
| Kontrak bulanan / 1 tahun / 2 tahun | 42,7% / 11,3% / 2,8% |
| Tenure 0–12 / 13–24 / 25–48 / 49–72 bulan | 47,4% / 28,7% / 20,4% / 9,5% |
| Fiber optic / DSL / tanpa internet | 41,9% / 19,0% / 7,4% |
| Electronic check / pembayaran otomatis | 45,3% / 15,2–16,7% |

- Kontrak bulanan mencakup 55% pelanggan tetapi **88,6% dari seluruh churn**.
- Pelanggan tahun pertama menyumbang **55,5%** churn.
- Hubungan beberapa fitur tidak linear (mis. churn memuncak pada 2 layanan tambahan lalu turun), dan `tenure` berkorelasi 0,83 dengan `TotalCharges` (multikolinearitas).

## Hasil modeling

**Validation set** (tiap model pada threshold optimalnya masing-masing):

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Interpretability |
|---|---|---|---|---|---|---|
| Dummy (baseline acak) | 0,496 | 0,264 | 0,259 | 0,257 | 0,258 | n/a |
| Logistic Regression (tuned) | 0,848 | 0,635 | 0,651 | 0,593 | 0,621 | Tinggi |
| Random Forest (tuned) | 0,848 | 0,644 | 0,608 | 0,636 | 0,621 | Sedang |
| **XGBoost (tuned)** | **0,851** | **0,652** | 0,601 | **0,671** | **0,634** | Rendah–Sedang |

**Test set (model final, threshold 0,41, dievaluasi satu kali):**

| ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| 0,843 | 0,671 | 0,626 | 0,594 | 0,610 |

Confusion matrix test: **TN 676, FP 100, FN 114, TP 167** (267 pelanggan dihubungi, 25,3%).

| Top-k% pelanggan berisiko tertinggi (test) | Precision | Recall | Lift |
|---|---|---|---|
| 10% | 0,783 | 0,295 | 2,9× |
| 20% | 0,664 | 0,498 | 2,5× |
| 30% | 0,587 | 0,662 | 2,2× |

**Keputusan penting**
- **Imbalance:** bobot kelas hampir tidak mengubah kualitas ranking (PR-AUC Logistic Regression 0,633 dengan dan tanpa bobot), hanya menggeser threshold efektif. Maka penanganan utamanya adalah **threshold tuning**. SMOTE tidak dipakai.
- **Tuning XGBoost:** PR-AUC CV 0,642 → 0,668, validation 0,635 → 0,652.
- **Threshold 0,41 (bukan 0,5):** dipilih di validation untuk memaksimalkan net benefit dengan batas kapasitas 30% pelanggan dihubungi.
- **Catatan statistik:** keunggulan XGBoost atas Logistic Regression **kecil dan tidak signifikan** (selisih PR-AUC +0,015, CI 95% bootstrap −0,010 sampai +0,042). Logistic Regression adalah alternatif transparan yang hampir setara.

## Interpretasi model

Tiga metode konsisten: SHAP, permutation importance, dan koefisien Logistic Regression.

- **Menaikkan risiko:** kontrak bulanan, internet fiber optic (odds ratio 3,08), electronic check, paperless billing.
- **Menurunkan risiko:** kontrak panjang, tenure lama, tanpa internet, OnlineSecurity dan TechSupport.
- **Fitur dengan |SHAP| tertinggi:** kontrak 1–2 tahun (0,41), `tenure` (0,32), fiber optic (0,27).
- `PriceChangeRatio` nyaris tidak berkontribusi, sehingga hipotesis *bill shock* tidak didukung data ini.

> Feature importance menunjukkan **kontribusi prediktif, bukan hubungan sebab-akibat.**

**Analisis kesalahan:** kesalahan paling berisiko adalah *false negative* (114 churn terlewat). Semua 38 churner berkontrak 1–2 tahun di test terlewat, dan *false negative rate* naik dari 24% (tenure 0–12 bulan) menjadi 85% (49–72 bulan).

## Dampak bisnis dan rekomendasi

**Asumsi kampanye (eksplisit dan dapat diubah):** biaya kontak 50 per pelanggan; peluang churn berhasil dipertahankan 30%; pendapatan diselamatkan = 6 bulan `MonthlyCharges`; kapasitas maksimum 30% pelanggan.

| Strategi (test, 1.057 pelanggan) | Dihubungi | Net benefit |
|---|---|---|
| **Model (threshold 0,41)** | 267 | **+9.916** (CI 95%: 7.394–12.523) |
| Acak (jumlah kontak sama) | 267 | −4.246 |
| Hubungi semua pelanggan | 1.057 | −16.810 |

Expected saved revenue 23.266, biaya kampanye 13.350. Program impas pada peluang berhasil **17,2%**.

**Rekomendasi**

| Temuan | Aksi | Estimasi dampak | Prioritas |
|---|---|---|---|
| Model memilah risiko dengan baik | Hubungi ±25% pelanggan teratas; mulai dengan uji A/B | Net ±9.916 per 1.057 pelanggan | High |
| Kontrak bulanan = 88,6% churn | Insentif migrasi ke kontrak 1 tahun | ±61 churn tercegah, ±24.300 pendapatan 6 bulan | High |
| Electronic check churn 45,3% vs 16,0% autopay | Kampanye autopay | ±52 churn tercegah, ±23.800 | High |
| Fiber tanpa TechSupport churn 49,4% | Bundling TechSupport, audit kualitas fiber | ±45 churn tercegah, ±23.500 | Medium |
| Tahun pertama = 55,5% churn | Program onboarding hari ke-30/60/90 | ±41 churn tercegah, ±13.800 | Medium |

Estimasi program memakai adopsi 10–20% dan diskon 50% untuk bias seleksi. Angkanya adalah **proyeksi pendapatan kotor, bukan laba**, dan tidak boleh dijumlahkan antar baris karena segmennya tumpang tindih.

## Keterbatasan

- Data observasional dan satu potret waktu; penyebab churn (kualitas jaringan, keluhan, kompetitor, harga) tidak tersedia, sehingga hasil berupa asosiasi.
- Model hampir tidak mendeteksi churner berkontrak panjang (blind spot).
- Test set hanya berisi 281 churn; recall turun dari 0,671 (validation) ke 0,594 (test) karena distribusi skor bergeser. Threshold berbasis kapasitas (top-k%) lebih stabil untuk produksi.
- Seluruh angka bisnis bergantung pada asumsi; uji A/B diperlukan untuk mengukur peluang berhasil dan biaya nyata.

## Struktur repositori

```
.
├── data/
│   └── 1A_telco_customer_churn.csv
├── Telco_Churn_Notebook.ipynb      # analisis lengkap (Task 1-4)
├── Telco_Churn_Notebook.html       # hasil export notebook yang sudah dijalankan
├── reports/
│   └── project_deck.pdf            # presentasi proyek
└── README.md
```

Sesuaikan dengan struktur folder yang benar-benar ada di repositori Anda.

## Cara menjalankan

**Prasyarat:** Python 3.10 atau lebih baru (notebook diuji pada Python 3.13).

```bash
git clone https://github.com/arielsharonferdinandus/Telco_Customer_Churn_Prediction.git
cd Telco_Customer_Churn_Prediction

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install pandas numpy scipy scikit-learn xgboost shap matplotlib seaborn jupyter nbconvert
jupyter notebook Telco_Churn_Notebook.ipynb
```

Jalankan **Kernel → Restart & Run All**. Notebook membaca `data/1A_telco_customer_churn.csv`; bila file tidak ditemukan, notebook mengunduhnya dari GitHub.

Untuk mengekspor hasil ke HTML:

```bash
jupyter nbconvert --to html Telco_Churn_Notebook.ipynb
```

> Tuning Random Forest memakan waktu beberapa menit (40 kandidat × 5-fold).

## Reproducibility

- `random_state = 42` dipakai konsisten untuk split, cross-validation, model, dan bootstrap.
- Angka dapat bergeser sedikit antar mesin atau versi library (terutama Random Forest dan waktu training); kesimpulan tidak berubah.

## Penggunaan AI

Claude (Anthropic) digunakan untuk membantu penulisan kode notebook, draf narasi dan interpretasi, serta penyusunan presentasi dan dokumen pendukung. Hasilnya diverifikasi dengan menjalankan ulang notebook secara penuh, pemeriksaan leakage otomatis, dan pencocokan angka antara notebook dan deck.

_[Sesuaikan bagian ini dengan penggunaan AI yang sebenarnya.]_

## Penulis

- **Nama:** [Nama Lengkap]
- **NIM:** [NIM]
- **Kelas:** [Kelas]
