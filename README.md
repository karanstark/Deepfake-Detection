 🧠 Deepfake Detection Using CNN

This project implements a Convolutional Neural Network (CNN)-based model to detect deepfake images and videos.  
It uses TensorFlow and Keras to train and evaluate the model for **real vs fake** face classification.


 🚀 Project Overview

Deepfakes are AI-generated fake media that manipulate images or videos to replace one person's likeness with another's.  
This project aims to build a deep learning model capable of detecting such manipulations with high accuracy.

Key Highlights:
- Built using TensorFlow and Keras  
- Uses a CNN architecture for image classification  
- Includes preprocessing, training, and evaluation scripts  
- Modular and easy-to-modify code structure  

 📂 Folder Structure
 Deepfake-Detection/
│
├── model_training.py # Main script for training the CNN model
├── deepfake_cnn_model.h5 # Trained model file (not tracked in Git)
├── train_images/ # Folder containing training images
├── fake_cifake_preds.json # JSON file containing image labels
├── venv/ # Virtual environment (ignored in Git)
├── requirements.txt # Python dependencies
└── README.md # Project documentation


⚙️ Installation and Setup

 1️⃣ Clone this repository
```bash
git clone https://github.com/karanstark/Deepfake-Detection.git
cd Deepfake-Detection

🧩 Requirements
Main libraries used:
TensorFlow 2.20.0
Keras
NumPy
Pandas
scikit-learn
OpenCV
Matplotlib
All required packages are listed in requirements.txt.

📊 Results

After training, the model classifies images as real (0) or fake (1) with good accuracy.
You can modify hyperparameters, epochs, or CNN layers for improved performance.

💡 Future Improvements

Add support for video-based detection using frame extraction

Integrate a web interface for real-time testing

Use transfer learning (e.g., EfficientNet, XceptionNet) for better accuracy.

License:
This project is licensed under the MIT License.
