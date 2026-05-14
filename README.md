# 🎵 Analisis Sentimen Ulasan Spotify — Google Play Store

Proyek ini menganalisis sentimen ulasan pengguna aplikasi **Spotify** di Google Play Store menggunakan tiga pendekatan machine learning / deep learning yang berbeda. Klasifikasi bersifat **binary**: ulasan negatif (rating 1–2) vs. positif (rating 4–5), dengan rating 3 (netral) dibuang.

---

## 📋 Daftar Isi

- [Gambaran Proyek](#-gambaran-proyek)
- [Struktur Repositori](#-struktur-repositori)
- [Dataset](#-dataset)
- [Skema Model](#-skema-model)
- [Pipeline Preprocessing](#-pipeline-preprocessing)
- [Hasil](#-hasil)
- [Cara Penggunaan](#-cara-penggunaan)
- [Dependensi](#-dependensi)

---

## 🔍 Gambaran Proyek

| Atribut | Detail |
|---|---|
| **Task** | Binary Sentiment Classification |
| **Domain** | Review Aplikasi — Google Play Store |
| **Bahasa Data** | Bahasa Indonesia (informal / slang) |
| **Sumber Data** | `google-play-scraper` (Spotify, `com.spotify.music`) |
| **Label** | `negatif` (rating 1–2) · `positif` (rating 4–5) |
| **Target Akurasi** | ≥ 85% |

---

## 📁 Struktur Repositori

```
├── Sentiment_Analysis_Spotify_Binary.ipynb   # Notebook utama (EDA + Training + Inference)
├── scraping_spotify_playstore.py             # Script scraping ulasan dari Play Store
├── ulasan_sp.csv                             # Dataset mentah hasil scraping
├── requirements.txt                          # Dependensi Python
└── README.md
```

Output yang dihasilkan setelah menjalankan notebook:

```
├── dataset_spotify_labeled.csv              # Dataset berlabel (setelah preprocessing)
├── model_svm_tfidf.pkl                      # Model LinearSVC + TF-IDF
├── model_bilstm_spotify.keras               # Model Bi-LSTM
├── keras_tokenizer.pkl                      # Tokenizer Keras (untuk Bi-LSTM)
├── model_word2vec_spotify.model             # Model Word2Vec
├── distribusi_rating_spotify.png
├── distribusi_sentimen_spotify.png
├── cm_skema1.png / cm_skema2.png / cm_skema3.png
└── perbandingan_skema.png
```

---

## 📦 Dataset

Data dikumpulkan menggunakan library `google-play-scraper` dengan konfigurasi:

- **Aplikasi**: Spotify (`com.spotify.music`)
- **Bahasa / Negara**: `id` / `id`
- **Sorting**: `MOST_RELEVANT`
- **Target pengambilan**: hingga 100.000 ulasan

Kolom utama yang digunakan: `score` (rating bintang) dan `content` (teks ulasan).

### Distribusi Label (setelah balancing)

| Label | Rating | Jumlah |
|---|---|---|
| `negatif` | 1–2 ⭐ | seimbang (≥ 3.500) |
| `positif` | 4–5 ⭐ | seimbang (≥ 3.500) |
| *(dibuang)* | 3 ⭐ | — |

Balancing dilakukan menggunakan `resample` (undersampling / oversampling ke kelas terkecil, minimal 3.500 per kelas).

---

## 🤖 Skema Model

### Skema 1 — TF-IDF + LinearSVC

| Komponen | Detail |
|---|---|
| **Fitur** | TF-IDF word n-gram (1,3) + TF-IDF char n-gram |
| **Classifier** | `LinearSVC` dengan regularisasi kuat |
| **Split** | 80% train / 20% test |
| **Evaluasi** | Accuracy, Classification Report, Confusion Matrix |

### Skema 2 — Bi-LSTM

| Komponen | Detail |
|---|---|
| **Fitur** | Embedding layer (trainable) |
| **Arsitektur** | `Embedding → SpatialDropout → Bidirectional LSTM → GlobalMaxPooling → Dense` |
| **Regularisasi** | Dropout agresif, EarlyStopping, ReduceLROnPlateau |
| **Split** | 80% train / 20% test |

### Skema 3 — Word2Vec + Random Forest

| Komponen | Detail |
|---|---|
| **Fitur** | Word2Vec (Skip-gram, `vector_size=200, window=5, epochs=20`) → rata-rata vektor per ulasan |
| **Classifier** | `RandomForestClassifier` (300 trees, `class_weight='balanced'`) |
| **Imbalance handling** | SMOTE pada data training |
| **Split** | 70% train / 30% test |

---

## 🧹 Pipeline Preprocessing

```
Teks mentah
  ↓  Cleaning (hapus URL, mention, emoji, angka, tanda baca)
  ↓  Normalisasi slang (kamus ~60+ entri, konteks Indonesia + Spotify)
  ↓  Negation handling ("tidak bagus" → token "bagus_NEG")
  ↓  Tokenisasi (NLTK word_tokenize)
  ↓  Stopword removal (Sastrawi + NLTK English, dengan whitelist kata sentimen)
  ↓  Tanpa stemming (stemming terbukti merusak kata sentimen pendek)
Teks bersih
```

**Catatan penting**: Stemming Sastrawi sengaja *tidak* digunakan karena menyebabkan kehilangan informasi sentimen pada ulasan pendek di Play Store.

---

## 📊 Hasil

Perbandingan performa ketiga skema (nilai aktual bergantung pada dataset saat runtime):

| Skema | Algoritma | Split | Train (%) | Test (%) | Status |
|---|---|---|---|---|---|
| Skema 1 | TF-IDF + LinearSVC | 80/20 | — | — | Target ≥ 85% |
| Skema 2 | Bi-LSTM | 80/20 | — | — | Target ≥ 85% |
| Skema 3 | Word2Vec + RF + SMOTE | 70/30 | — | — | Target ≥ 85% |

> Jalankan notebook untuk melihat angka akurasi aktual pada dataset Anda.

---

## 🚀 Cara Penggunaan

### 1. Clone repositori

```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>
```

### 2. Install dependensi utama

```bash
pip install google-play-scraper sastrawi gensim nltk scikit-learn tensorflow wordcloud matplotlib seaborn imbalanced-learn
```

> Atau gunakan `requirements.txt` (environment Google Colab lengkap):
> ```bash
> pip install -r requirements.txt
> ```

### 3. Scraping data (opsional — skip jika sudah ada `ulasan_sp.csv`)

```bash
python scraping_spotify_playstore.py
```

### 4. Jalankan notebook

Buka `Sentiment_Analysis_Spotify_Binary.ipynb` di Google Colab atau Jupyter, lalu jalankan sel secara berurutan.

### 5. Inference

Gunakan fungsi `predict_bilstm()` atau `predict_svm()` di bagian akhir notebook untuk memprediksi sentimen ulasan baru secara interaktif.

---

## 🛠️ Dependensi Utama

| Library | Kegunaan |
|---|---|
| `google-play-scraper` | Scraping ulasan Play Store |
| `Sastrawi` | Stopword removal Bahasa Indonesia |
| `NLTK` | Tokenisasi, stopwords |
| `scikit-learn` | TF-IDF, LinearSVC, Random Forest, metrics |
| `TensorFlow / Keras` | Bi-LSTM |
| `Gensim` | Word2Vec |
| `imbalanced-learn` | SMOTE |
| `pandas`, `numpy` | Manipulasi data |
| `matplotlib`, `seaborn`, `wordcloud` | Visualisasi |

---

## 👤 Author

**Rifqi Zaghlul Musyaffa**  
Mahasiswa Matematika — Universitas Brawijaya  
AI Engineer Path — Coding Camp 2026 powered by DBS Foundation (Dicoding)

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan pembelajaran dalam program Coding Camp 2026. Bebas digunakan sebagai referensi dengan menyertakan atribusi.
