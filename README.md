# Face_Recognition_NCNN_CPP

📌 **Realtime Face Recognition using NCNN and C++ on Linux**

This project implements a real-time face recognition system using [Tencent's NCNN framework](https://github.com/Tencent/ncnn) with pre-trained models like RetinaFace and ArcFace. The system detects faces from webcam, aligns them, extracts embeddings, and compares them to determine identity in real-time.

---

## 📁 Folder Structure

```
Face_Recognition_Ncnn_Linux/
├── build/                  # Empty folder (ignored in Git)
├── input/                 # Input images
│   └── capture/           # Captured images from webcam
├── models/                # (Place pre-trained models here - see below)
├── ncnn/                  # Cloned NCNN repo (built locally)
├── output/                # Results: images & embeddings
│   └── embedding/         # Saved embedding vectors (512D)
├── src/                   # C++ source files
└── CMakeLists.txt         # Build instructions
```

---

## 📦 Download Pretrained Models

Download the required models from the following Google Drive folder:

🔗 **[Download Models (Google Drive)](https://drive.google.com/drive/folders/19Jj8nPhX0BK_Xen6y05aog0bdJs1k2QO?usp=sharing)**

Place the downloaded files into the `models/` directory:

- `arcface.param`
- `arcface.bin`
- `mnet.25-opt.param`
- `mnet.25-opt.bin`

---

## ⚙️ Components & Features

### 1. **Face Detection (RetinaFace)**
- Detects faces in image or webcam using `mnet.25-opt` model.
- Uses NCNN for fast inference.
- Applies Non-Maximum Suppression (NMS) with IoU threshold `0.4`.
- Extracts 5-point facial landmarks for alignment.

### 2. **Face Alignment**
- Aligns detected faces to `112x112` size using affine transformation from landmarks.

### 3. **Embedding Extraction (ArcFace)**
- Uses ArcFace model to extract a 512-dimensional vector per face.
- Embedding vectors are saved as `.txt`.

### 4. **Face Recognition**
- Cosine similarity is used to compare two embeddings.
- If similarity ≥ `0.5`, same person.
- Real-time recognition from camera stream with label overlay.

---

## 🧱 Build Instructions

### 🔧 Requirements
- Linux system
- CMake
- OpenCV (>= 3.4)
- NCNN (already cloned in `ncnn/`)

### 📥 Build steps

```bash
# Clone your own repository (after uploading to GitHub)
git clone https://github.com/your_username/Face_Recognition_NCNN_CPP.git
cd Face_Recognition_NCNN_CPP

# Create build directory
mkdir build && cd build

# Build using CMake
cmake ..
make
```

---

## ▶️ Usage

### 1. **Detect & Align Faces**
```bash
./main ../input/face1.jpg
```

### 2. **Extract Embedding**
```bash
./embedding ../output/face1.jpg ../output/embedding/face1_embedding.txt
```

### 3. **Compare Two Embeddings**
```bash
./compare_embedding ../output/embedding/face1_embedding.txt ../output/embedding/face2_embedding.txt
```

### 4. **Real-Time Face Recognition**
```bash
./recognition ../output/embedding/face1_embedding.txt 0
```
- Replace `0` with the appropriate webcam index if needed.

---

## 📌 Source File Overview

| File | Description |
|------|-------------|
| `main.cpp` | Detect face and align using RetinaFace |
| `embedding.cpp` | Extract 512D embedding from aligned face |
| `compare_embedding.cpp` | Compare two embeddings using cosine similarity |
| `capture.cpp` | Capture image from webcam |
| `recognition.cpp` | Real-time recognition from webcam |

---

## 🧠 Acknowledgements

- [Tencent NCNN](https://github.com/Tencent/ncnn) - Lightweight neural network inference framework.
- [InsightFace](https://github.com/deepinsight/insightface) - Face detection & recognition models.
