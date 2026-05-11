# Install dan Training dengan Docker

Dokumen ini menjelaskan langkah dari nol sampai dashboard Matcha Sentiment berjalan. Semua command diasumsikan dijalankan dari root project.

## 1. Prasyarat

Install komponen berikut:

| Kebutuhan | Fungsi |
| --- | --- |
| Docker Desktop | Build image dan menjalankan dashboard/training |
| NVIDIA Driver | Akses GPU laptop dari Docker |
| WSL2 backend Docker | Runtime Linux untuk container CUDA |
| Git | Clone/push repo |
| Hugging Face token | Deploy ke Hugging Face Spaces, opsional |

Cek GPU dari host:

```powershell
nvidia-smi
```

Cek Docker:

```powershell
docker --version
docker info
```

Pastikan `docker info` menampilkan runtime NVIDIA atau Docker Desktop sudah mengaktifkan GPU support.

## 2. Clone atau Buka Folder Project

```powershell
cd "D:\matcha sentiment"
```

Kalau dari GitHub:

```powershell
git clone <repo-url>
cd <repo-folder>
```

## 3. Build Docker Image

Image memakai PyTorch CUDA runtime, jadi ukuran build pertama cukup besar.

```powershell
docker build -t matcha-sentiment .
```

Base image di `Dockerfile`:

```dockerfile
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
```

PyTorch 2.6+ dipakai agar aman dan kompatibel saat memuat checkpoint Transformer `.bin` dari Hugging Face.

## 4. Cek GPU di Dalam Container

```powershell
docker run --rm --gpus all -v "${PWD}:/workspace" matcha-sentiment python scripts/check_gpu.py
```

Output yang diharapkan mirip:

```json
{
  "torch_version": "2.6.0+cu124",
  "cuda_available": true,
  "cuda_device_count": 1,
  "device_name": "NVIDIA GeForce RTX 3070 Ti Laptop GPU"
}
```

Kalau `cuda_available` masih `false`, cek Docker Desktop, WSL2, NVIDIA driver, dan coba restart Docker Desktop.

## 5. Siapkan Dataset Binary

Script ini membaca Excel asli, menghapus label `Netral`, membersihkan teks kosong, dan membuang duplikat.

```powershell
docker run --rm -v "${PWD}:/workspace" matcha-sentiment python scripts/prepare_data.py
```

Output:

```text
data/processed/matcha_sentiment_binary.csv
data/processed/summary.json
```

Dataset final run terakhir:

| Label | Jumlah |
| --- | ---: |
| Negatif | 371 |
| Positif | 326 |
| Total | 697 |

## 6. Training Machine Learning Klasik

Pipeline klasik melatih:

- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- TF-IDF + Random Forest
- TF-IDF + Extra Trees
- TF-IDF + Gradient Boosting
- Word2Vec + Logistic Regression
- Word2Vec + Linear SVM
- Word2Vec + Random Forest
- Word2Vec + Extra Trees
- Word2Vec + Gradient Boosting

Jalankan 10-fold validation:

```powershell
docker run --rm -v "${PWD}:/workspace" matcha-sentiment python scripts/train_classical.py --folds 10
```

Output penting:

```text
artifacts/classical/results.csv
artifacts/classical/fold_metrics.csv
artifacts/classical/keyword_counts.csv
artifacts/classical/top_words_tfidf.csv
models/classical/best_model.joblib
artifacts/figures/classical_best_confusion_matrix.png
artifacts/figures/classical_best_roc_auc.png
artifacts/figures/top_words_tfidf.png
artifacts/figures/wordcloud_positif.png
artifacts/figures/wordcloud_negatif.png
```

## 7. Training 5 Transformer Indonesia

Default model:

```text
indobenchmark/indobert-base-p1
indobenchmark/indobert-base-p2
indolem/indobert-base-uncased
indolem/indobertweet-base-uncased
flax-community/indonesian-roberta-base
```

Jalankan training dengan GPU wajib:

```powershell
docker run --rm --gpus all -v "${PWD}:/workspace" matcha-sentiment python scripts/train_transformers.py --require-gpu
```

Untuk GPU 8 GB, default `batch-size=8` dan `max-length=160` sudah dicoba. Kalau VRAM penuh:

```powershell
docker run --rm --gpus all -v "${PWD}:/workspace" matcha-sentiment python scripts/train_transformers.py --require-gpu --batch-size 4 --eval-batch-size 8
```

Output penting:

```text
artifacts/transformers/results.csv
artifacts/transformers/results.json
models/transformers/<model-slug>/model
models/best_transformer
artifacts/figures/transformer_best_training_loss.png
artifacts/figures/transformer_best_confusion_matrix.png
artifacts/figures/transformer_best_roc_auc.png
```

## 8. Jalankan Semua Pipeline Sekaligus

```powershell
docker run --rm --gpus all -v "${PWD}:/workspace" matcha-sentiment python scripts/train_all.py --require-gpu
```

Dengan Docker Compose:

```powershell
docker compose run --rm train
```

## 9. Jalankan Dashboard

```powershell
docker run --rm --gpus all -p 7860:7860 -v "${PWD}:/workspace" --name matcha-sentiment-app matcha-sentiment
```

Buka:

```text
http://localhost:7860
```

Stop dashboard:

```powershell
docker rm -f matcha-sentiment-app
```

## 10. Smoke Test API

Saat dashboard aktif:

```powershell
docker exec matcha-sentiment-app python -c "from gradio_client import Client; c=Client('http://127.0.0.1:7860'); print(c.predict('Matchanya enak dan pelayanannya ramah.', api_name='/predict_review')); print(c.predict('Harganya mahal dan pelayanannya lama.', api_name='/predict_review'))"
```

Contoh hasil:

```text
Positif - transformer on cuda
Negatif - transformer on cuda
```

## 11. Screenshot dan Visual untuk GitHub

Screenshot dan plot siap push ada di:

```text
docs/images/
```

File utama:

```text
dashboard_home_wide.png
dashboard_visual_tab.png
dashboard_keywords_tab.png
results_gallery.png
transformer_best_training_loss.png
transformer_best_confusion_matrix.png
transformer_best_roc_auc.png
top_words_tfidf.png
wordcloud_positif.png
wordcloud_negatif.png
```

## 12. Deploy ke Hugging Face Spaces

Login:

```powershell
hf auth login
```

Buat Space Docker:

```powershell
hf repo create USERNAME/matcha-sentiment --repo-type space --space_sdk docker --exist-ok
```

Upload:

```powershell
hf upload USERNAME/matcha-sentiment . --repo-type space --exclude "*.xlsx" --exclude ".cache/*" --exclude "__pycache__/*" --exclude "models/transformers/*" --exclude "artifacts/transformers/*/checkpoints/*"
```

Kalau `hf` belum ada di host, gunakan CLI dalam Docker:

```powershell
docker run --rm -it -v "${PWD}:/workspace" -e HF_TOKEN=hf_xxx matcha-sentiment hf repo create USERNAME/matcha-sentiment --repo-type space --space_sdk docker --exist-ok
docker run --rm -it -v "${PWD}:/workspace" -e HF_TOKEN=hf_xxx matcha-sentiment hf upload USERNAME/matcha-sentiment . --repo-type space --exclude "*.xlsx" --exclude ".cache/*" --exclude "__pycache__/*" --exclude "models/transformers/*" --exclude "artifacts/transformers/*/checkpoints/*"
```

Untuk inference saja, CPU Space bisa jalan tetapi lebih lambat. Kalau ingin prediksi lebih responsif, pilih GPU di Settings Hugging Face Space.

## 13. Push ke GitHub

Model terbaik berukuran sekitar 442 MB:

```text
models/best_transformer/model.safetensors
```

GitHub biasa membatasi file besar, jadi gunakan Git LFS kalau model ikut dipush.

```powershell
git lfs install
git lfs track "*.safetensors"
git lfs track "*.bin"
git lfs track "models/**"
```

Pastikan `.gitattributes` ikut masuk commit:

```powershell
git add .gitattributes
```

Checklist file yang sebaiknya ikut commit:

```text
README.md
INSTALL_DOCKER.md
Dockerfile
docker-compose.yml
requirements.txt
app.py
scripts/
src/
data/processed/
artifacts/classical/
artifacts/transformers/results.csv
artifacts/transformers/results.json
artifacts/figures/
docs/images/
models/best_transformer/
models/classical/
```

Commit:

```powershell
git add README.md INSTALL_DOCKER.md Dockerfile docker-compose.yml requirements.txt app.py scripts src data/processed artifacts docs models .gitignore .gitattributes
git commit -m "Add matcha sentiment training dashboard"
git push origin main
```

Kalau tidak ingin upload model besar ke GitHub, hapus `models/best_transformer/` dari commit dan deploy model ke Hugging Face Model Hub terpisah.

## 14. Troubleshooting

### CUDA tidak terbaca

Jalankan:

```powershell
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Kalau command kedua gagal, masalah ada di Docker Desktop atau NVIDIA Container Toolkit support.

### Out of memory saat training Transformer

Turunkan batch size:

```powershell
docker run --rm --gpus all -v "${PWD}:/workspace" matcha-sentiment python scripts/train_transformers.py --require-gpu --batch-size 4 --eval-batch-size 8
```

### Port 7860 sudah dipakai

Gunakan port lain:

```powershell
docker run --rm --gpus all -p 7861:7860 -v "${PWD}:/workspace" matcha-sentiment
```

Buka:

```text
http://localhost:7861
```

### Model belum muncul di dashboard

Pastikan file ini ada:

```text
models/best_transformer/config.json
models/best_transformer/model.safetensors
models/best_transformer/tokenizer.json
```

Kalau belum ada, jalankan lagi training Transformer.
