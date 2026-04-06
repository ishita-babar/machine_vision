# Adaptive Document Enhancement using Machine Vision

A full-stack Machine Vision based Document Enhancement System that improves scanned or photographed documents using adaptive image processing techniques.

The system automatically selects the best enhancement pipeline and shows step-by-step processing outputs.

---

## Features

- Upload document image  
- Automatic model-based enhancement decision  
- Multi-stage image processing pipeline  
- Step-by-step visualization  
- Final enhanced document output  
- Web-based UI (React + Flask)

---

## Tech Stack

### Frontend
- React
- JavaScript
- HTML / CSS

### Backend
- Flask
- Python
- OpenCV
- NumPy

### Machine Learning
- Trained model (`model.pkl`)
- Feature extraction based adaptive processing

---

## Project Structure

```
machine_vision/
│
├── backend/
│   ├── app.py
│   ├── processing.py
│   ├── feature_extraction.py
│   ├── model/
│   │   └── model.pkl
│   └── temp/
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│
├── dataset/
│   └── scan_doc_rotation
│
├── notebooks/
│   ├── train.py
│   └── training.ipynb
│
├── utils/
├── requirements.txt
├── run.py
└── README.md
```

---

## Processing Pipeline

The system performs the following steps:

- Original Image  
- Deskew  
- Grayscale Conversion  
- Flatten  
- Denoise  
- CLAHE Enhancement  
- Pre Threshold  
- Threshold  
- Final Output  

---

## Installation

### Clone Repository

```
git clone https://github.com/ishita-babar/machine_vision.git
cd machine_vision
```

---

## Backend Setup

Install dependencies:

```
pip install -r requirements.txt
```

Run backend:

```
python -u run.py
```

---

## Frontend Setup

Open another terminal:

```
cd frontend
npm install
npm start
```

Frontend runs on:

```
http://localhost:3000
```

Backend runs on:

```
http://localhost:5000
```

---

## Model

- Model file: `model/model.pkl`
- Used for adaptive pipeline selection
- Based on extracted document features

---

## Dataset

Dataset used for training available in:

```
dataset/scan_doc_rotation
```

---

## Requirements

- Python 3.8+
- Node.js
- npm
- Flask
- OpenCV
- NumPy
- React

---

## Authors

Aditya Deshmukh - 23BAI0072
Aditya Agarwal - 23BAI0211
Ishita Babar - 23BAI0080
Nakul Kamdar - 23BAI0120
