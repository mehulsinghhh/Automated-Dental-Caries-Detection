# Automated-Dental-Caries-Detection
🦷 Automated Dental Caries Detection (YOLOv5 + CNN + OpenCV)

This project focuses on automated detection of dental caries (tooth cavities) from dental X-ray images using Deep Learning. The aim is to assist dentists by providing a fast, reliable, and reproducible diagnostic support system.

🚀 Project Overview

Dental caries detection from X-ray images is often challenging due to variations in lighting, overlapping structures, and subtle lesion visibility.
This project uses a combination of:

YOLOv5 for object detection (detecting caries regions in real time)

Convolutional Neural Networks (CNN) for classification of detected regions

OpenCV for image preprocessing and visualization

Random Forest for auxiliary classification/decision support

Python + TensorFlow + PyTorch

The system processes dental X-rays, identifies suspicious carious lesions, and highlights them with bounding boxes.

🧠 Tech Stack

Python

PyTorch (YOLOv5)

TensorFlow/Keras (CNN model)

OpenCV

NumPy, Pandas

Random Forest Classifier (for feature-based classification)

Matplotlib / Seaborn (visualization)

⚙️ Features

✔️ Preprocessing of dental X-ray images (denoising, contrast enhancement)
✔️ Real-time caries detection using YOLOv5
✔️ Hybrid pipeline: YOLOv5 + CNN + Random Forest
✔️ Bounding box generation and visual highlight of caries
✔️ Model evaluation (accuracy, precision, recall, F1-score)
✔️ Easy-to-run training and inference scripts

📂 Project Structure (Example)
├── data/
│   ├── images/
│   └── labels/
├── models/
│   ├── yolo_model.pt
│   └── cnn_model.h5
├── src/
│   ├── preprocess.py
│   ├── detect_yolo.py
│   ├── cnn_classifier.py
│   ├── random_forest.py
│   └── utils.py
├── results/
├── README.md
└── requirements.txt

🛠️ How It Works

Image Preprocessing
Enhances quality using histogram equalization, resizing, thresholding.

YOLOv5 Detection
Detects potential caries regions on X-ray images in real time.

CNN Classification
Each detected ROIs (Regions of Interest) is passed through a CNN model to classify whether it is a carious lesion.

Random Forest Model
Helps in combining extracted features for improved classification reliability.

Output
The final result highlights caries on X-ray images with bounding boxes.

📊 Results (Example)

YOLOv5 Detection Accuracy: ~90%

CNN Classification Accuracy: ~92%

Combined Pipeline F1 Score: ~0.88
