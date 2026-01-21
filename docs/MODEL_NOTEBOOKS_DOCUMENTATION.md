# 📚 Dokumentasi Model Notebooks - Autism Classifier

Dokumen ini menjelaskan dua notebook training yang digunakan dalam project Autism Classifier.

---

## 📋 Daftar Isi

1. [Overview](#overview)
2. [Notebook 1: Train by Qullah (PyTorch)](#notebook-1-train-by-qullah-pytorch)
3. [Notebook 2: Train MTCNN (TensorFlow/Keras)](#notebook-2-train-mtcnn-tensorflowkeras)
4. [Perbandingan Kedua Model](#perbandingan-kedua-model)
5. [Kesimpulan](#kesimpulan)

---

## Overview

Project ini memiliki **dua pendekatan** untuk klasifikasi autisme berdasarkan gambar wajah:

| Aspek              | Train by Qullah             | Train MTCNN      |
| ------------------ | --------------------------- | ---------------- |
| **Framework**      | PyTorch                     | TensorFlow/Keras |
| **Face Detection** | Tidak ada (resize langsung) | MTCNN            |
| **Model Output**   | `.pth`                      | `.h5`            |
| **GPU Support**    | CUDA                        | TensorFlow GPU   |

---

## Notebook 1: Train by Qullah (PyTorch)

### 📁 File: `train by qullah.ipynb`

### 1.1 Import Libraries

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
```

**Penjelasan:**

- `torch` - Library utama PyTorch untuk deep learning
- `torch.nn` - Module untuk membangun neural network
- `torch.optim` - Optimizer seperti Adam, SGD
- `DataLoader` - Untuk batch processing data
- `transforms` - Untuk data augmentation

### 1.2 Konfigurasi

```python
size = (224, 224)  # Ukuran input gambar
classes = ['Autistic', 'Non_Autistic']  # Label kelas
```

**Kenapa 224x224?**

- Standar untuk CNN (ImageNet standard)
- Balance antara detail dan komputasi
- Kompatibel dengan banyak pretrained model

### 1.3 Loading Data

```python
def ld_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(cv2.resize(img, size), cv2.COLOR_BGR2RGB)

    folder_name = os.path.basename(os.path.dirname(path))
    label = classes.index(folder_name)

    return img / 255.0, label  # Normalisasi ke 0-1
```

**Penjelasan:**

1. Baca gambar dengan OpenCV
2. Resize ke 224x224
3. Convert BGR → RGB (OpenCV default adalah BGR)
4. Normalisasi pixel values dari 0-255 ke 0-1
5. Label berdasarkan nama folder

```python
# Convert ke format PyTorch: (N, C, H, W)
images = np.array(images).transpose(0, 3, 1, 2)  # NHWC -> NCHW
```

**Format Tensor:**

- **NHWC** = (Batch, Height, Width, Channels) - TensorFlow format
- **NCHW** = (Batch, Channels, Height, Width) - PyTorch format

### 1.4 Data Augmentation

```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),    # Flip horizontal 50%
    transforms.RandomRotation(10),              # Rotasi ±10 derajat
    transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),  # Zoom 90-110%
])
```

**Kenapa Augmentation?**

- Mencegah overfitting
- Meningkatkan generalisasi model
- Memperbanyak variasi data training

### 1.5 Arsitektur Model CNN

```python
class AutismCNN(nn.Module):
    def __init__(self):
        super(AutismCNN, self).__init__()

        # Block 1: 3 → 32 channels
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 224 → 112
            nn.Dropout(0.25)
        )

        # Block 2: 32 → 64 channels
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 112 → 56
            nn.Dropout(0.25)
        )

        # Block 3: 64 → 128 channels
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 56 → 28
            nn.Dropout(0.25)
        )

        # Block 4: 128 → 256 channels
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 28 → 14
            nn.Dropout(0.4)
        )

        # Classifier (Fully Connected)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),  # 50176 → 512
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()          # Output 0-1 untuk binary classification
        )
```

**Penjelasan Layer:**

| Layer         | Fungsi                                |
| ------------- | ------------------------------------- |
| `Conv2d`      | Ekstraksi fitur dari gambar           |
| `BatchNorm2d` | Normalisasi untuk stabilitas training |
| `ReLU`        | Activation function (non-linear)      |
| `MaxPool2d`   | Reduce dimensi, ambil fitur dominan   |
| `Dropout`     | Regularisasi, cegah overfitting       |
| `Flatten`     | Convert 2D → 1D untuk dense layer     |
| `Linear`      | Fully connected layer                 |
| `Sigmoid`     | Output probability 0-1                |

**Visualisasi Dimensi:**

```
Input:  (batch, 3, 224, 224)
        ↓ Block1
        (batch, 32, 112, 112)
        ↓ Block2
        (batch, 64, 56, 56)
        ↓ Block3
        (batch, 128, 28, 28)
        ↓ Block4
        (batch, 256, 14, 14)
        ↓ Flatten
        (batch, 50176)
        ↓ Dense layers
Output: (batch, 1) → probability
```

### 1.6 Training Configuration

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()  # Binary Cross Entropy
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',      # Minimize val_loss
    factor=0.5,      # Kurangi LR jadi setengah
    patience=5,      # Tunggu 5 epoch sebelum reduce
    min_lr=1e-7      # Minimum LR
)
```

**Penjelasan:**

- **Adam Optimizer**: Adaptive learning rate, bagus untuk kebanyakan kasus
- **BCELoss**: Binary Cross Entropy untuk klasifikasi 2 kelas
- **ReduceLROnPlateau**: Auto-reduce learning rate jika val_loss tidak turun

### 1.7 Training Loop

```python
for epoch in range(start_epoch, epochs):
    # === TRAINING PHASE ===
    model.train()  # Set mode training (dropout aktif)
    for batch_x, batch_y in train_loader:
        batch_x = augment_batch(batch_x)  # Apply augmentation

        optimizer.zero_grad()              # Reset gradients
        outputs = model(batch_x).squeeze() # Forward pass
        loss = criterion(outputs, batch_y) # Hitung loss
        loss.backward()                    # Backpropagation
        optimizer.step()                   # Update weights

    # === VALIDATION PHASE ===
    model.eval()  # Set mode eval (dropout non-aktif)
    with torch.no_grad():  # Tidak perlu gradients
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x).squeeze()
            # Hitung val_loss dan val_acc
```

**Alur Training:**

1. `model.train()` → Aktifkan dropout & batch norm training mode
2. `optimizer.zero_grad()` → Reset gradients dari iterasi sebelumnya
3. Forward pass → Prediksi output
4. Hitung loss → Bandingkan prediksi dengan label asli
5. `loss.backward()` → Hitung gradients (backpropagation)
6. `optimizer.step()` → Update weights berdasarkan gradients

### 1.8 Checkpoint & Resume Training

```python
# Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_acc': best_val_acc,
    'history': history,
}, CHECKPOINT_FULL_PATH)

# Load checkpoint
checkpoint = torch.load(CHECKPOINT_FULL_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

**Keuntungan:**

- Resume training dari titik terakhir
- Tidak kehilangan progress jika training terhenti
- Backup otomatis setiap N epoch

---

## Notebook 2: Train MTCNN (TensorFlow/Keras)

### 📁 File: `train (Mtcnn).ipynb`

### 2.1 Import Libraries

```python
from tensorflow import keras
from tensorflow.keras import layers
from mtcnn import MTCNN
```

**Penjelasan:**

- `keras` - High-level API untuk TensorFlow
- `MTCNN` - Multi-task Cascaded Convolutional Networks untuk face detection

### 2.2 MTCNN Face Detection

```python
detector = MTCNN()

def align_face(image, keypoints):
    """Align face berdasarkan posisi mata"""
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']

    # Hitung sudut rotasi
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # Rotasi gambar
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def crop_face(image, box, margin=0.3):
    """Crop wajah dengan margin"""
    x, y, w, h = box
    size = int(max(w, h) * (1 + margin))  # Tambah margin 30%
    # ... crop logic
    return image[y1:y2, x1:x2]
```

**Alur MTCNN:**

1. Detect wajah → Dapatkan bounding box & keypoints
2. Align face → Rotasi agar mata horizontal
3. Crop face → Ambil area wajah saja dengan margin

**Keypoints yang dideteksi:**

- `left_eye`, `right_eye`
- `nose`
- `mouth_left`, `mouth_right`

### 2.3 Arsitektur Model (Keras)

```python
model = keras.Sequential([
    layers.Input(shape=(224, 224, 3)),

    # Data Augmentation (built-in)
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),

    # Conv layers
    layers.Conv2D(24, 3, activation='relu'),
    layers.MaxPooling2D(2),
    layers.Conv2D(32, 2, activation='relu'),
    layers.MaxPooling2D(2),
    layers.Conv2D(64, 1, activation='relu'),
    layers.MaxPooling2D(2),

    # Classifier
    layers.Flatten(),
    layers.Dense(1, activation='sigmoid')
])
```

**Perbedaan dengan PyTorch model:**

- Lebih sederhana (3 conv layers vs 8)
- Tidak ada BatchNorm
- Tidak ada Dropout
- Augmentation di dalam model

### 2.4 Training (Keras Style)

```python
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=[early_stop]
)
```

**Kelebihan Keras:**

- API lebih simpel
- Callbacks built-in (EarlyStopping, ModelCheckpoint, dll)
- Training dalam satu line (`model.fit()`)

---

## Perbandingan Kedua Model

### Tabel Perbandingan

| Aspek               | PyTorch (Qullah) | TensorFlow (MTCNN) |
| ------------------- | ---------------- | ------------------ |
| **Framework**       | PyTorch          | TensorFlow/Keras   |
| **Face Detection**  | ❌ Tidak ada     | ✅ MTCNN           |
| **Face Alignment**  | ❌ Tidak ada     | ✅ Ya              |
| **Conv Layers**     | 8 layers         | 3 layers           |
| **BatchNorm**       | ✅ Ya            | ❌ Tidak           |
| **Dropout**         | ✅ Ya (0.25-0.5) | ❌ Tidak           |
| **Parameters**      | ~26M             | ~500K              |
| **Resume Training** | ✅ Ya            | ❌ Tidak           |
| **Backup System**   | ✅ Ya            | ❌ Tidak           |
| **GPU Support**     | CUDA             | TensorFlow GPU     |

### Kapan Menggunakan Masing-masing?

**Gunakan PyTorch (Qullah) jika:**

- Butuh model yang lebih dalam dan robust
- Ingin resume training
- Dataset sudah di-crop wajahnya
- Butuh kontrol penuh atas training

**Gunakan TensorFlow (MTCNN) jika:**

- Gambar input bukan hanya wajah
- Butuh face alignment otomatis
- Ingin training cepat dengan API simpel
- Deployment ke TensorFlow Serving

### Interpretasi Output

**PyTorch Model:**

```python
# Output: probability untuk Non_Autistic
prediction = model(image).item()
is_autistic = prediction < 0.5   # < 0.5 = Autistic
confidence = 1 - prediction if is_autistic else prediction
```

**TensorFlow Model:**

```python
# Output: probability untuk Autistic
prediction = model.predict(image)[0][0]
is_autistic = prediction >= 0.5  # >= 0.5 = Autistic
confidence = prediction if is_autistic else 1 - prediction
```

⚠️ **Perhatian:** Kedua model memiliki interpretasi output yang berbeda!

---

## Kesimpulan

### Best Practices yang Dipelajari:

1. **Data Preprocessing**
   - Normalisasi pixel values (0-255 → 0-1)
   - Resize ke ukuran standar (224x224)
   - Augmentation untuk generalisasi

2. **Model Architecture**
   - BatchNorm untuk stabilitas
   - Dropout untuk regularisasi
   - Progressive channel increase (32→64→128→256)

3. **Training**
   - Learning rate scheduling
   - Early stopping
   - Checkpoint & backup

4. **Face Detection**
   - MTCNN untuk gambar non-cropped
   - Face alignment meningkatkan akurasi

### File Output

| File                             | Deskripsi                    |
| -------------------------------- | ---------------------------- |
| `autism_cnn_model_by_qullah.pth` | Best PyTorch weights         |
| `autism_cnn_checkpoint.pth`      | Full checkpoint untuk resume |
| `autism_cnn_model(MTCNN).h5`     | TensorFlow model             |
| `backups/*.pth`                  | Backup berkala               |

---

## Referensi

- [PyTorch Documentation](https://pytorch.org/docs/)
- [TensorFlow/Keras Guide](https://www.tensorflow.org/guide)
- [MTCNN Paper](https://arxiv.org/abs/1604.02878)
- [Understanding CNNs](https://cs231n.github.io/convolutional-networks/)

---

_Dokumentasi ini dibuat untuk keperluan pembelajaran. Last updated: January 2026_
