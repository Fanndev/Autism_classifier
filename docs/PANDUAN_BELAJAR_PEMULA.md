# 🎓 Panduan Belajar Deep Learning untuk Pemula Total

Dokumentasi ini dibuat khusus untuk pemula yang ingin memahami project Autism Classifier dari NOL.

---

## 📋 Daftar Isi

1. [Persiapan Awal](#1-persiapan-awal)
2. [Konsep Dasar Python](#2-konsep-dasar-python)
3. [Pengenalan Machine Learning](#3-pengenalan-machine-learning)
4. [Pengenalan Deep Learning & CNN](#4-pengenalan-deep-learning--cnn)
5. [Penjelasan Kode Line-by-Line](#5-penjelasan-kode-line-by-line)
6. [Cara Menjalankan Project](#6-cara-menjalankan-project)
7. [Troubleshooting](#7-troubleshooting)
8. [Glosarium](#8-glosarium)

---

## 1. Persiapan Awal

### 1.1 Software yang Dibutuhkan

| Software      | Fungsi               | Download                                               |
| ------------- | -------------------- | ------------------------------------------------------ |
| Python 3.10+  | Bahasa pemrograman   | [python.org](https://python.org)                       |
| VS Code       | Code editor          | [code.visualstudio.com](https://code.visualstudio.com) |
| Git           | Version control      | [git-scm.com](https://git-scm.com)                     |
| NVIDIA Driver | Untuk GPU (opsional) | [nvidia.com](https://nvidia.com/drivers)               |

### 1.2 Struktur Folder Project

```
Autism_classifier/
├── 📁 datasets/           # Data gambar untuk training
│   ├── train/            # Data latih (70%)
│   │   ├── Autistic/     # Gambar anak autis
│   │   └── Non_Autistic/ # Gambar anak non-autis
│   ├── valid/            # Data validasi (15%)
│   └── test/             # Data test (15%)
│
├── 📁 notebooks/          # Jupyter notebooks untuk training
│   ├── train by qullah.ipynb    # Training dengan PyTorch
│   ├── train (Mtcnn).ipynb      # Training dengan TensorFlow
│   └── model/                    # Model hasil training
│       ├── autism_cnn_model_by_qullah.pth
│       └── autism_cnn_model(MTCNN).h5
│
├── 📁 classifier/         # Django app untuk web
│   ├── views.py          # Handle HTTP request
│   ├── forms.py          # Form upload gambar
│   └── infrastructure/
│       └── predictor.py  # Load model & prediksi
│
├── 📁 templates/          # HTML templates
├── 📁 static/             # CSS, JS files
├── 📁 docs/               # Dokumentasi
│
├── manage.py             # Django management
├── requirements.txt      # Daftar library
└── README.md
```

### 1.3 Install Dependencies

```bash
# Buat virtual environment
python -m venv .venv

# Aktifkan virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install semua library
pip install -r requirements.txt
```

---

## 2. Konsep Dasar Python

### 2.1 Variables & Data Types

```python
# String - teks
nama = "Autism Classifier"

# Integer - bilangan bulat
epochs = 50

# Float - bilangan desimal
learning_rate = 0.001

# Boolean - True/False
is_training = True

# List - kumpulan data
classes = ['Autistic', 'Non_Autistic']

# Dictionary - key-value pairs
config = {
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001
}

# Tuple - list yang tidak bisa diubah
size = (224, 224)
```

### 2.2 Functions

```python
# Definisi function
def hitung_akurasi(correct, total):
    """
    Menghitung akurasi prediksi.

    Args:
        correct: Jumlah prediksi benar
        total: Total data

    Returns:
        Akurasi dalam bentuk desimal (0-1)
    """
    return correct / total

# Memanggil function
akurasi = hitung_akurasi(85, 100)  # Output: 0.85
print(f"Akurasi: {akurasi * 100}%")  # Output: Akurasi: 85.0%
```

### 2.3 Classes (OOP)

```python
# Class adalah blueprint/template untuk membuat object
class Hewan:
    def __init__(self, nama, kaki):
        self.nama = nama    # Attribute
        self.kaki = kaki

    def bersuara(self):     # Method
        print(f"{self.nama} bersuara!")

# Membuat object dari class
kucing = Hewan("Kucing", 4)
kucing.bersuara()  # Output: Kucing bersuara!
```

### 2.4 Import Library

```python
# Import seluruh module
import numpy as np

# Import function tertentu
from os.path import join, exists

# Import dengan alias
import matplotlib.pyplot as plt

# Import class dari module
from torch.utils.data import DataLoader
```

### 2.5 List Comprehension

```python
# Cara biasa
squares = []
for i in range(5):
    squares.append(i ** 2)
# Result: [0, 1, 4, 9, 16]

# Dengan list comprehension (lebih singkat)
squares = [i ** 2 for i in range(5)]
# Result: [0, 1, 4, 9, 16]
```

---

## 3. Pengenalan Machine Learning

### 3.1 Apa itu Machine Learning?

**Machine Learning (ML)** adalah cabang AI dimana komputer **belajar dari data** tanpa diprogram secara eksplisit.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    DATA     │ --> │   MODEL     │ --> │  PREDIKSI   │
│  (Gambar)   │     │   (CNN)     │     │ (Autistic?) │
└─────────────┘     └─────────────┘     └─────────────┘
```

### 3.2 Jenis Machine Learning

| Jenis             | Penjelasan                     | Contoh             |
| ----------------- | ------------------------------ | ------------------ |
| **Supervised**    | Belajar dari data berlabel     | Klasifikasi gambar |
| **Unsupervised**  | Belajar dari data tanpa label  | Clustering         |
| **Reinforcement** | Belajar dari reward/punishment | Game AI            |

Project ini menggunakan **Supervised Learning** karena:

- Kita punya gambar (input)
- Kita punya label: Autistic atau Non_Autistic (output)

### 3.3 Workflow Machine Learning

```
1. COLLECT DATA
   ↓
2. PREPROCESS (bersihkan, resize, normalisasi)
   ↓
3. SPLIT DATA (train/validation/test)
   ↓
4. BUILD MODEL (arsitektur neural network)
   ↓
5. TRAIN MODEL (belajar dari data)
   ↓
6. EVALUATE (ukur performa)
   ↓
7. DEPLOY (gunakan untuk prediksi)
```

### 3.4 Train, Validation, Test Split

```
Total Data: 100%
├── Train (70%): Data untuk melatih model
├── Validation (15%): Data untuk tuning hyperparameter
└── Test (15%): Data untuk evaluasi final

Kenapa dipisah?
- Train: Model belajar dari sini
- Validation: Cek apakah model overfitting
- Test: Evaluasi jujur (model belum pernah lihat)
```

### 3.5 Overfitting vs Underfitting

```
UNDERFITTING                    GOOD FIT                      OVERFITTING
(Model terlalu simple)          (Model pas)                   (Model terlalu complex)

     ╭────╮                      ╭──────╮                      ╭╮ ╭╮ ╭╮
    ╱      ╲                    ╱        ╲                    ╱╰╯╰╯╰╯╲
───╱────────╲───          ─────╱──────────╲─────         ───╱──────────╲───
  •  •  •  •              •  •  •  •  •  •              •  •  •  •  •  •

Train Acc: 60%                Train Acc: 85%                Train Acc: 99%
Val Acc: 58%                  Val Acc: 83%                  Val Acc: 70%

PROBLEM: Tidak belajar        IDEAL: Generalisasi baik      PROBLEM: Hafal data train
                                                            tapi gagal di data baru
```

---

## 4. Pengenalan Deep Learning & CNN

### 4.1 Apa itu Neural Network?

Neural Network terinspirasi dari otak manusia:

```
INPUT LAYER          HIDDEN LAYERS           OUTPUT LAYER
(Pixel gambar)       (Ekstraksi fitur)       (Prediksi)

    ○                    ○
    ○───────────────────○○                      ○ Autistic
    ○                    ○○────────────────────
    ○───────────────────○○                      ○ Non_Autistic
    ○                    ○
```

### 4.2 Bagaimana Neural Network Belajar?

```
1. FORWARD PASS
   Input → Model → Prediksi

2. HITUNG LOSS (ERROR)
   Loss = Prediksi - Label Asli

3. BACKPROPAGATION
   Hitung gradients (seberapa besar error dari tiap weight)

4. UPDATE WEIGHTS
   Weights baru = Weights lama - (learning_rate × gradient)

5. ULANGI sampai loss kecil
```

**Analogi:**

> Seperti belajar memanah. Pertama tembak (forward), lihat meleset kemana (loss),
> sesuaikan posisi (backprop), tembak lagi. Ulangi sampai kena sasaran.

### 4.3 Apa itu CNN (Convolutional Neural Network)?

CNN adalah neural network khusus untuk **memproses gambar**.

```
INPUT          CONVOLUTION        POOLING         FULLY CONNECTED    OUTPUT
(Gambar)       (Deteksi fitur)    (Reduce size)   (Klasifikasi)

┌─────────┐    ┌─────────┐       ┌─────┐         ┌───┐
│░░░░░░░░░│    │ Edge    │       │     │         │   │───○ Autistic
│░░███░░░░│ -> │ Corner  │   ->  │     │    ->   │   │
│░░░░░░░░░│    │ Texture │       │     │         │   │───○ Non_Autistic
└─────────┘    └─────────┘       └─────┘         └───┘
  224x224       112x112           56x56           512
```

### 4.4 Layer-Layer dalam CNN

#### 4.4.1 Convolutional Layer (Conv2D)

**Fungsi:** Mendeteksi pola/fitur dalam gambar

```
FILTER/KERNEL (3x3)         GAMBAR INPUT              OUTPUT (Feature Map)
┌───┬───┬───┐               ┌───┬───┬───┬───┐
│-1 │ 0 │ 1 │               │ 1 │ 2 │ 3 │ 4 │        Hasil konvolusi:
├───┼───┼───┤      *        ├───┼───┼───┼───┤   =    Deteksi edge vertikal
│-1 │ 0 │ 1 │               │ 5 │ 6 │ 7 │ 8 │
├───┼───┼───┤               ├───┼───┼───┼───┤
│-1 │ 0 │ 1 │               │ 9 │10 │11 │12 │
└───┴───┴───┘               └───┴───┴───┴───┘
```

```python
nn.Conv2d(
    in_channels=3,      # RGB = 3 channel
    out_channels=32,    # Buat 32 filter berbeda
    kernel_size=3,      # Filter ukuran 3x3
    padding=1           # Tambah border agar ukuran tetap
)
```

#### 4.4.2 Batch Normalization

**Fungsi:** Menstabilkan dan mempercepat training

```python
nn.BatchNorm2d(32)  # Normalize 32 channels

# Apa yang dilakukan:
# 1. Hitung mean dan std dari batch
# 2. Normalize: (x - mean) / std
# 3. Scale dan shift: gamma * x_norm + beta
```

**Kenapa perlu?**

- Mencegah "internal covariate shift"
- Training lebih stabil
- Bisa pakai learning rate lebih besar

#### 4.4.3 Activation Function (ReLU)

**Fungsi:** Menambahkan non-linearity

```
ReLU (Rectified Linear Unit):
f(x) = max(0, x)

Input:  [-2, -1, 0, 1, 2, 3]
Output: [ 0,  0, 0, 1, 2, 3]  # Negative jadi 0

        ▲ output
        │      ╱
        │     ╱
        │    ╱
────────┼───╱──────▶ input
        │  ╱
        │ ╱
        │╱
```

**Kenapa ReLU?**

- Simple dan cepat
- Tidak ada vanishing gradient problem
- Sparse activation (banyak 0)

#### 4.4.4 Max Pooling

**Fungsi:** Reduce ukuran, ambil fitur paling penting

```
INPUT (4x4)                    OUTPUT (2x2)
┌────┬────┬────┬────┐          ┌────┬────┐
│ 1  │ 3  │ 2  │ 1  │          │    │    │
├────┼────┼────┼────┤   Max    │ 4  │ 6  │
│ 4  │ 2  │ 6  │ 4  │  ───▶   ├────┼────┤
├────┼────┼────┼────┤  Pool    │    │    │
│ 3  │ 1  │ 2  │ 3  │  (2x2)   │ 5  │ 8  │
├────┼────┼────┼────┤          └────┴────┘
│ 5  │ 2  │ 8  │ 1  │
└────┴────┴────┴────┘

Ambil nilai MAX dari setiap region 2x2
```

```python
nn.MaxPool2d(2)  # Pool size 2x2, stride 2
# 224x224 → 112x112 (ukuran jadi setengah)
```

#### 4.4.5 Dropout

**Fungsi:** Regularisasi untuk mencegah overfitting

```
TRAINING (Dropout 0.5)              INFERENCE (Semua aktif)
○───○───○───○───○                   ○───○───○───○───○
│   │   │   │   │                   │   │   │   │   │
○───X───○───X───○   Random          ○───○───○───○───○
│   │   │   │   │   matikan         │   │   │   │   │
○───○───X───○───X   50% neuron      ○───○───○───○───○

X = neuron dimatikan
```

```python
nn.Dropout(0.25)  # Matikan 25% neuron secara random
```

**Kenapa mencegah overfitting?**

- Model tidak bisa bergantung pada neuron tertentu
- Seperti "ensemble" banyak model kecil

#### 4.4.6 Fully Connected (Linear/Dense)

**Fungsi:** Klasifikasi berdasarkan fitur yang diekstrak

```python
nn.Linear(
    in_features=50176,  # Input: 256 × 14 × 14 = 50176
    out_features=512    # Output: 512 neurons
)
```

#### 4.4.7 Sigmoid

**Fungsi:** Convert output ke probability (0-1)

```
Sigmoid: f(x) = 1 / (1 + e^(-x))

Input:  [-3, -1, 0, 1, 3]
Output: [0.05, 0.27, 0.5, 0.73, 0.95]

         ▲ output
       1 │        ────────
         │      ╱
     0.5 │─────●
         │    ╱
       0 │────
         └──────────────▶ input
```

### 4.5 Loss Function: Binary Cross Entropy

```python
criterion = nn.BCELoss()
```

**Rumus:**

```
BCE = -[y × log(p) + (1-y) × log(1-p)]

Dimana:
- y = label asli (0 atau 1)
- p = prediksi model (0 sampai 1)
```

**Contoh:**

```
Prediksi: 0.9, Label: 1 (Autistic)
BCE = -[1 × log(0.9) + 0 × log(0.1)]
BCE = -log(0.9) = 0.105  # Loss kecil (bagus!)

Prediksi: 0.9, Label: 0 (Non_Autistic)
BCE = -[0 × log(0.9) + 1 × log(0.1)]
BCE = -log(0.1) = 2.303  # Loss besar (salah!)
```

### 4.6 Optimizer: Adam

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

**Apa itu Optimizer?**

- Algoritma untuk update weights berdasarkan gradients
- Adam = Adaptive Moment Estimation
- Kombinasi Momentum + RMSprop

**Learning Rate:**

```
learning_rate = 0.001

Weights_baru = Weights_lama - learning_rate × gradient

LR terlalu besar → Overshoot, tidak converge
LR terlalu kecil → Training sangat lambat
LR pas → Converge dengan baik
```

### 4.7 Learning Rate Scheduler

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',      # Monitor nilai yang mau diminimize
    factor=0.5,      # Kurangi LR jadi setengah
    patience=5,      # Tunggu 5 epoch sebelum reduce
    min_lr=1e-7      # Batas minimum LR
)
```

**Cara kerja:**

```
Epoch 1-5:   LR = 0.001
Epoch 6-10:  LR = 0.001 (val_loss masih turun)
Epoch 11-15: LR = 0.001 (val_loss stuck)
Epoch 16:    LR = 0.0005 (dikurangi karena 5 epoch tidak improve)
...
```

---

## 5. Penjelasan Kode Line-by-Line

### 5.1 Cell 1: Import Libraries

```python
import glob                    # Untuk mencari file dengan pattern
import numpy as np             # Numerical computing (array, matrix)
import pandas as pd            # Data manipulation (tabel)
import seaborn as sns          # Visualisasi statistik
import matplotlib.pyplot as plt # Plotting grafik
import cv2                     # OpenCV - image processing
import torch                   # PyTorch - deep learning framework
import torch.nn as nn          # Neural network modules
import torch.optim as optim    # Optimizer (Adam, SGD, dll)
from torch.utils.data import DataLoader, TensorDataset  # Batch data
import torchvision.transforms as transforms  # Image augmentation
from sklearn.metrics import confusion_matrix, classification_report
from concurrent.futures import ThreadPoolExecutor  # Parallel processing
import os                      # Operating system functions
from tqdm import tqdm          # Progress bar
from rich import print         # Pretty printing
```

### 5.2 Cell 1: GPU Detection

```python
# Cek apakah GPU tersedia
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # Gunakan GPU
    device = torch.device('cuda:0')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    # Fallback ke CPU
    device = torch.device('cpu')
    print("Menggunakan CPU")
```

**Penjelasan:**

- CUDA = API NVIDIA untuk komputasi GPU
- Training di GPU 10-100x lebih cepat dari CPU
- `cuda:0` = GPU pertama

### 5.3 Cell 2: Konfigurasi

```python
size = (224, 224)                    # Ukuran input gambar
classes = ['Autistic', 'Non_Autistic']  # Label kelas
```

**Kenapa 224×224?**

- Standar ImageNet
- Cukup detail untuk wajah
- Tidak terlalu berat untuk komputasi

### 5.4 Cell 3: Load Image Function

```python
def ld_img(path):
    # 1. Baca gambar dari file
    img = cv2.imread(path)

    # 2. Cek apakah gambar berhasil dibaca
    if img is None:
        raise ValueError(f"Failed to load image: {path}")

    # 3. Resize gambar ke 224x224
    # 4. Convert BGR ke RGB (OpenCV default adalah BGR)
    img = cv2.cvtColor(cv2.resize(img, size), cv2.COLOR_BGR2RGB)

    # 5. Ambil nama folder sebagai label
    folder_name = os.path.basename(os.path.dirname(path))
    # Contoh: path = "datasets/train/Autistic/img001.jpg"
    #         folder_name = "Autistic"

    # 6. Convert label ke angka (index)
    label = classes.index(folder_name)
    # "Autistic" → 0
    # "Non_Autistic" → 1

    # 7. Normalisasi pixel dari 0-255 ke 0-1
    return img / 255.0, label
```

**Kenapa normalisasi?**

```
Sebelum: pixel values 0-255
Sesudah: pixel values 0-1

Manfaat:
- Gradient tidak explode
- Training lebih stabil
- Convergence lebih cepat
```

### 5.5 Cell 3: Parallel Loading

```python
def ld(folder_path, max_workers=8):
    # 1. Cari semua file gambar (jpg, png)
    paths = glob.glob(
        os.path.join(folder_path, '**', '*.[jp][pn][g]'),
        recursive=True
    )
    # Pattern: *.[jp][pn][g] matches .jpg, .png, .jpeg

    # 2. Load gambar secara parallel (8 thread sekaligus)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(ld_img, paths),
            total=len(paths),
            desc="Read"
        ))

    # 3. Pisahkan images dan labels
    images, labels = zip(*results)

    # 4. Convert ke PyTorch format
    # Dari: (N, H, W, C) - NHWC
    # Ke:   (N, C, H, W) - NCHW (PyTorch format)
    images = np.array(images).transpose(0, 3, 1, 2)

    return torch.tensor(images, dtype=torch.float32), \
           torch.tensor(labels, dtype=torch.float32)
```

**Parallel Processing:**

```
Sequential (1 thread):
Image1 → Image2 → Image3 → Image4
[====]   [====]   [====]   [====]
Total: 4 detik

Parallel (4 threads):
Image1 [====]
Image2 [====]
Image3 [====]
Image4 [====]
Total: 1 detik
```

### 5.6 Cell 8: Data Augmentation

```python
train_transform = transforms.Compose([
    # 1. Flip horizontal dengan probabilitas 50%
    transforms.RandomHorizontalFlip(p=0.5),

    # 2. Rotasi random ±10 derajat
    transforms.RandomRotation(10),

    # 3. Zoom random 90-110%
    transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
])

def augment_batch(images):
    """Apply augmentation ke batch images"""
    augmented = []
    for img in images:
        aug_img = train_transform(img)
        augmented.append(aug_img)
    return torch.stack(augmented)
```

**Visualisasi Augmentation:**

```
ORIGINAL        FLIP           ROTATE         ZOOM IN
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  ◕  ◕  │    │  ◕  ◕  │    │ ◕    ◕ │    │         │
│    👃   │ → │   👃    │    │   👃    │    │  ◕  ◕  │
│   ───   │    │   ───   │    │  ───    │    │    👃   │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
```

### 5.7 Cell 11: Model Architecture

```python
class AutismCNN(nn.Module):
    def __init__(self):
        super(AutismCNN, self).__init__()

        # ============ BLOCK 1 ============
        # Input: (batch, 3, 224, 224)
        self.block1 = nn.Sequential(
            # Conv: 3 → 32 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            # Output: (batch, 32, 224, 224)

            nn.BatchNorm2d(32),  # Normalize
            nn.ReLU(),           # Activation

            # Conv lagi: 32 → 32 channels
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Reduce size: 224 → 112
            nn.MaxPool2d(2),
            # Output: (batch, 32, 112, 112)

            nn.Dropout(0.25)  # Matikan 25% neuron
        )

        # ============ BLOCK 2 ============
        # Input: (batch, 32, 112, 112)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112 → 56
            nn.Dropout(0.25)
        )
        # Output: (batch, 64, 56, 56)

        # ============ BLOCK 3 ============
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56 → 28
            nn.Dropout(0.25)
        )
        # Output: (batch, 128, 28, 28)

        # ============ BLOCK 4 ============
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 → 14
            nn.Dropout(0.4)   # Dropout lebih besar
        )
        # Output: (batch, 256, 14, 14)

        # ============ CLASSIFIER ============
        self.classifier = nn.Sequential(
            # Flatten: (batch, 256, 14, 14) → (batch, 50176)
            nn.Flatten(),

            # Dense: 50176 → 512
            nn.Linear(256 * 14 * 14, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            # Dense: 512 → 128
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Output: 128 → 1
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output 0-1
        )

    def forward(self, x):
        """Forward pass - data mengalir dari input ke output"""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x
```

**Visualisasi Dimensi:**

```
Layer               Output Shape         Parameters
────────────────────────────────────────────────────
Input               (batch, 3, 224, 224)     0
Block1              (batch, 32, 112, 112)    10,400
Block2              (batch, 64, 56, 56)      55,552
Block3              (batch, 128, 28, 28)     221,824
Block4              (batch, 256, 14, 14)     295,424
Flatten             (batch, 50176)           0
Dense1              (batch, 512)             25,690,624
Dense2              (batch, 128)             65,664
Output              (batch, 1)               129
────────────────────────────────────────────────────
Total                                        ~26 Million
```

### 5.8 Cell 13: Training Loop

```python
for epoch in range(start_epoch, epochs):

    # ========== TRAINING PHASE ==========
    model.train()  # Set mode training
    # Dropout AKTIF, BatchNorm update statistics

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_x, batch_y in train_loader:
        # 1. Apply augmentation
        batch_x = augment_batch(batch_x)

        # 2. Reset gradients
        optimizer.zero_grad()
        # Penting! Kalau tidak di-reset, gradient akan terakumulasi

        # 3. Forward pass
        outputs = model(batch_x).squeeze()
        # squeeze(): (batch, 1) → (batch,)

        # 4. Hitung loss
        loss = criterion(outputs, batch_y)

        # 5. Backpropagation
        loss.backward()
        # Hitung gradient untuk setiap parameter

        # 6. Update weights
        optimizer.step()
        # weights = weights - learning_rate * gradient

        # 7. Track metrics
        train_loss += loss.item()
        predicted = (outputs > 0.5).float()  # Threshold 0.5
        train_correct += (predicted == batch_y).sum().item()
        train_total += batch_y.size(0)

    # Hitung average loss dan accuracy
    train_acc = train_correct / train_total
    train_loss = train_loss / len(train_loader)

    # ========== VALIDATION PHASE ==========
    model.eval()  # Set mode evaluation
    # Dropout NON-AKTIF, BatchNorm pakai saved statistics

    with torch.no_grad():  # Tidak perlu hitung gradient
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)

            val_loss += loss.item()
            predicted = (outputs > 0.5).float()
            val_correct += (predicted == batch_y).sum().item()

    val_acc = val_correct / val_total

    # ========== LEARNING RATE SCHEDULING ==========
    scheduler.step(val_loss)
    # Jika val_loss tidak turun selama 5 epoch,
    # kurangi learning rate jadi setengah

    # ========== SAVE BEST MODEL ==========
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f'Best model saved! val_acc: {best_val_acc:.4f}')

    # ========== EARLY STOPPING ==========
    if patience_counter >= patience:
        print('Early stopping!')
        break
```

**Alur Training Satu Epoch:**

```
┌─────────────────────────────────────────────────────────────┐
│                      TRAINING PHASE                          │
├─────────────────────────────────────────────────────────────┤
│  Batch 1    Batch 2    Batch 3    ...    Batch N            │
│  [====]     [====]     [====]            [====]             │
│     ↓          ↓          ↓                 ↓               │
│  Forward   Forward    Forward           Forward             │
│     ↓          ↓          ↓                 ↓               │
│  Loss      Loss       Loss              Loss                │
│     ↓          ↓          ↓                 ↓               │
│  Backward  Backward   Backward          Backward            │
│     ↓          ↓          ↓                 ↓               │
│  Update    Update     Update            Update              │
├─────────────────────────────────────────────────────────────┤
│                     VALIDATION PHASE                         │
├─────────────────────────────────────────────────────────────┤
│  Batch 1    Batch 2    Batch 3    ...    Batch M            │
│  [====]     [====]     [====]            [====]             │
│     ↓          ↓          ↓                 ↓               │
│  Forward   Forward    Forward           Forward             │
│     ↓          ↓          ↓                 ↓               │
│  Evaluate  Evaluate   Evaluate          Evaluate            │
│                                                             │
│  → Calculate val_loss dan val_accuracy                      │
│  → Update learning rate scheduler                           │
│  → Save model jika best                                     │
│  → Check early stopping                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Cara Menjalankan Project

### 6.1 Setup Environment

```bash
# 1. Clone repository (jika dari Git)
git clone <repository-url>
cd Autism_classifier

# 2. Buat virtual environment
python -m venv .venv

# 3. Aktifkan virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Jika pakai GPU, install PyTorch dengan CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 6.2 Training Model

```bash
# 1. Buka VS Code
code .

# 2. Buka notebook
# notebooks/train by qullah.ipynb

# 3. Pilih kernel Python dari .venv

# 4. Run All Cells
# Ctrl+Shift+Enter atau klik "Run All"
```

### 6.3 Menjalankan Website

```bash
# 1. Aktifkan virtual environment
.venv\Scripts\activate

# 2. Jalankan Django server
python manage.py runserver

# 3. Buka browser
# http://127.0.0.1:8000
```

### 6.4 Menggunakan Website

```
1. Buka http://127.0.0.1:8000
2. Pilih model dari dropdown
3. Upload gambar wajah
4. Klik "Analisis"
5. Lihat hasil prediksi
```

---

## 7. Troubleshooting

### 7.1 CUDA Not Available

**Problem:**

```
torch.cuda.is_available() = False
```

**Solusi:**

```bash
# 1. Cek NVIDIA driver
nvidia-smi

# 2. Install PyTorch dengan CUDA yang sesuai
# Cek versi CUDA dari nvidia-smi
# Lalu install:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3. Restart Python/Notebook
```

### 7.2 Out of Memory (OOM)

**Problem:**

```
CUDA out of memory
```

**Solusi:**

```python
# 1. Kurangi batch size
batch_size = 16  # dari 32

# 2. Atau gunakan gradient accumulation
accumulation_steps = 2
for i, (batch_x, batch_y) in enumerate(train_loader):
    loss = criterion(model(batch_x), batch_y)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 7.3 Overfitting

**Problem:**

```
Train accuracy: 99%
Validation accuracy: 70%
```

**Solusi:**

```python
# 1. Tambah dropout
nn.Dropout(0.5)  # dari 0.25

# 2. Tambah augmentation
transforms.RandomHorizontalFlip(p=0.5),
transforms.RandomRotation(15),  # dari 10
transforms.ColorJitter(brightness=0.2, contrast=0.2),

# 3. Tambah weight decay (L2 regularization)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 4. Kurangi kompleksitas model
# Kurangi jumlah layer atau neurons
```

### 7.4 Model Tidak Converge

**Problem:**

```
Loss tidak turun atau naik turun tidak stabil
```

**Solusi:**

```python
# 1. Kurangi learning rate
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # dari 0.001

# 2. Gunakan learning rate warmup
from torch.optim.lr_scheduler import LinearLR
warmup = LinearLR(optimizer, start_factor=0.1, total_iters=5)

# 3. Cek data (mungkin ada yang corrupt)
# 4. Cek label (mungkin tertukar)
```

### 7.5 Import Error

**Problem:**

```
ModuleNotFoundError: No module named 'torch'
```

**Solusi:**

```bash
# 1. Pastikan virtual environment aktif
.venv\Scripts\activate

# 2. Install ulang
pip install torch

# 3. Cek apakah terinstall
pip list | grep torch
```

---

## 8. Glosarium

| Term                    | Penjelasan                                          |
| ----------------------- | --------------------------------------------------- |
| **Activation Function** | Fungsi non-linear untuk menambah kompleksitas model |
| **Backpropagation**     | Algoritma untuk menghitung gradient                 |
| **Batch**               | Subset data yang diproses bersamaan                 |
| **Batch Size**          | Jumlah sample dalam satu batch                      |
| **CNN**                 | Convolutional Neural Network                        |
| **Convolution**         | Operasi sliding filter untuk ekstraksi fitur        |
| **CUDA**                | Platform komputasi GPU dari NVIDIA                  |
| **Dropout**             | Teknik regularisasi dengan mematikan neuron random  |
| **Epoch**               | Satu kali iterasi seluruh dataset                   |
| **Feature Map**         | Output dari convolutional layer                     |
| **Forward Pass**        | Aliran data dari input ke output                    |
| **Gradient**            | Turunan loss terhadap parameter                     |
| **Gradient Descent**    | Algoritma optimasi untuk minimize loss              |
| **Hyperparameter**      | Parameter yang diset sebelum training               |
| **Kernel/Filter**       | Matrix kecil untuk operasi konvolusi                |
| **Learning Rate**       | Seberapa besar langkah update weights               |
| **Loss Function**       | Fungsi untuk mengukur error prediksi                |
| **Max Pooling**         | Operasi downsampling dengan mengambil nilai max     |
| **Normalization**       | Scaling data ke range tertentu (0-1)                |
| **Optimizer**           | Algoritma untuk update weights                      |
| **Overfitting**         | Model terlalu "hafal" data training                 |
| **Parameter**           | Weights dan biases dalam model                      |
| **Pooling**             | Operasi untuk reduce dimensi                        |
| **ReLU**                | Rectified Linear Unit activation                    |
| **Regularization**      | Teknik untuk mencegah overfitting                   |
| **Sigmoid**             | Fungsi yang output 0-1                              |
| **Tensor**              | Array multi-dimensi (generalisasi matrix)           |
| **Training**            | Proses model belajar dari data                      |
| **Underfitting**        | Model terlalu simple untuk data                     |
| **Validation**          | Evaluasi model pada data yang tidak ditraining      |
| **Weight**              | Parameter yang dipelajari oleh model                |

---

## 🎯 Tips Belajar

1. **Hands-on Practice**
   - Jangan hanya baca, langsung coba coding
   - Ubah parameter, lihat hasilnya

2. **Start Simple**
   - Mulai dengan model kecil
   - Tambah kompleksitas bertahap

3. **Visualize Everything**
   - Plot loss, accuracy
   - Lihat sample predictions
   - Visualize feature maps

4. **Read Error Messages**
   - Error message biasanya informatif
   - Google error yang tidak dimengerti

5. **Join Community**
   - Stack Overflow
   - PyTorch Forums
   - Reddit r/MachineLearning

---

## 📚 Resources Tambahan

### Video Tutorials

- [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Sentdex - PyTorch Tutorial](https://www.youtube.com/playlist?list=PLQVvvaa0QuDdeMyHEYc0gxFpYwHY2Qfdh)

### Courses

- [Fast.ai - Practical Deep Learning](https://course.fast.ai/)
- [Coursera - Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)

### Documentation

- [PyTorch Official Tutorial](https://pytorch.org/tutorials/)
- [TensorFlow Guide](https://www.tensorflow.org/guide)

---

_Dokumentasi ini dibuat untuk pemula total. Jika ada yang kurang jelas, jangan ragu untuk bertanya!_

_Last updated: January 2026_
