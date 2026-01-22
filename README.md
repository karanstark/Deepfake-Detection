# 🧠 Deepfake Detection Using CNN

A **Convolutional Neural Network (CNN)** based deep learning project to detect **deepfake images and videos**.
This system classifies face images as **Real** or **Fake** using **TensorFlow** and **Keras**.

---

## 🚀 Project Overview

Deepfakes are AI-generated media where a person's face is manipulated or replaced using deep learning techniques. These fake visuals pose serious threats in misinformation, identity theft, and digital fraud.

This project focuses on building an effective CNN-based classifier capable of detecting such manipulated face images.

### ✨ Key Features

* CNN-based deep learning model
* Binary classification: **Real (0) vs Fake (1)**
* Built with TensorFlow & Keras
* Modular, clean, and beginner-friendly code
* Easy to extend for video-based detection

---

## 🛠️ Tech Stack

* **Python 3.x**
* **TensorFlow 2.20.0**
* **Keras**
* **NumPy**
* **Pandas**
* **scikit-learn**
* **OpenCV**
* **Matplotlib**

---

## 📂 Project Structure

```
Deepfake-Detection/
│
├── model_training.py          # CNN model training script
├── deepfake_cnn_model.h5      # Trained model file (ignored in Git)
├── train_images/              # Dataset directory (real & fake images)
├── fake_cifake_preds.json     # Image labels (ground truth)
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── venv/                      # Virtual environment (ignored in Git)
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/karanstark/Deepfake-Detection.git
cd Deepfake-Detection
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\\Scripts\\activate         # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Model Training

To train the CNN model, run:

```bash
python model_training.py
```

The trained model will be saved as:

```
deepfake_cnn_model.h5
```

---

## 📊 Results

* The model successfully classifies face images as **Real (0)** or **Fake (1)**
* Performance depends on dataset size, image quality, and training epochs
* Accuracy can be improved by tuning hyperparameters or using advanced architectures

---

## 🔮 Future Enhancements

* 🎥 **Video Deepfake Detection** using frame extraction
* 🧠 **Transfer Learning** with EfficientNet, XceptionNet, or ResNet
* 🌐 **Web Application** for real-time deepfake detection
* 📈 Improve performance using data augmentation & larger datasets

---

## 📜 License

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute this project with proper attribution.

---

## 👤 Author

**Karan Stark**
GitHub: https://github.com/karanstark)

---

⭐ If you like this project, don’t forget to **star the repository**!
