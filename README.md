# Autism Classifier Project

Proyek ini adalah sistem klasifikasi autisme berbasis Deep Learning yang diintegrasikan dengan aplikasi web Django. Sistem ini menggunakan Convolutional Neural Network (CNN) dengan PyTorch untuk mendeteksi indikasi autisme dari citra wajah.

## Fitur Utama

- **Model Deep Learning**: Dibangun menggunakan PyTorch dengan arsitektur CNN.
- **Deteksi Wajah**: Menggunakan MTCNN untuk cropping wajah yang presisi sebelum klasifikasi.
- **Aplikasi Web**: Antarmuka pengguna berbasis Django untuk mengunggah gambar dan melihat hasil prediksi.
- **Notebook Eksperimen**: Tersedia Jupyter Notebook untuk proses training, evaluasi, dan eksperimen model.

## Struktur Proyek

- `autism_site/`: Konfigurasi utama proyek Django.
- `classifier/`: Aplikasi Django yang menangani logika klasifikasi (Views, URLs).
- `datasets/`: Folder untuk dataset (Train, Valid, Test) gambar wajah.
- `model/`: Tempat penyimpanan model yang sudah dilatih (`.pth`).
- `notebooks/`: Jupyter Notebook untuk eksplorasi data dan training model.
- `static/`: File statis (CSS, JS, Images).
- `templates/`: Template HTML untuk aplikasi web.
- `manage.py`: Utilitas manajemen proyek Django.

## Prasyarat

- Python 3.10 atau lebih baru.
- NVIDIA GPU (Opsional, tapi sangat disarankan untuk training) dengan CUDA 12.6.

## Instalasi

1.  **Clone Repository** (atau ekstrak folder proyek):

    ```bash
    git clone https://github.com/Fanndev/Autism_classifier.git
    cd Autism_classifier
    ```

2.  **Buat Virtual Environment** (Rekomendasi):

    ```bash
    # Windows
    python -m venv .venv
    .venv\Scripts\activate

    # Linux/Mac
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Instal Dependencies**:
    Instal PyTorch terlebih dahulu (sesuaikan dengan versi CUDA Anda, contoh untuk CUDA 12.6):

    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    ```

    Kemudian instal sisa dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Siapkan Database**:
    ```bash
    python manage.py migrate
    ```

## Penggunaan

### 1. Menjalankan Aplikasi Web

Untuk memulai server pengembangan Django:

```bash
python manage.py runserver
```

Akses aplikasi melalui browser di `http://127.0.0.1:8000/`.

### 2. Melatih Model

Jika Anda ingin melatih ulang model:

1.  Pastikan dataset tersedia di folder `datasets/`.
2.  Buka notebook di folder `notebooks/`:
    ```bash
    jupyter notebook
    ```
3.  Jalankan `train by qullah.ipynb` untuk memulai training.
4.  Model terbaik akan disimpan di folder `model/`.

## Teknologi yang Digunakan

- **Backend & Web**: Django 5.x
- **Machine Learning**: PyTorch 2.x
- **Image Processing**: OpenCV, MTCNN, Pillow
- **Data Analysis**: Numpy, Pandas, Matplotlib, Seaborn

## Catatan

- Pastikan folder `model/` memiliki file model (`.pth`) yang valid agar aplikasi web dapat melakukan prediksi.
- Gunakan GPU (NVIDIA RTX series direkomendasikan) untuk mempercepat proses training.
