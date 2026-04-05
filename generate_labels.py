import cv2
import os
import csv
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(SCRIPT_DIR, "dataset", "scan_doc_rotation", "images")
output_csv = os.path.join(SCRIPT_DIR, "dataset", "labels.csv")


def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    noise = np.std(gray)
    return blur, brightness, noise


def decide_pipeline(blur, brightness, noise):
    if noise > 40:
        return "denoise+clahe+adaptive"
    elif blur < 50:
        return "strong_denoise+otsu"
    elif brightness < 100:
        return "clahe+adaptive"
    else:
        return "clahe"


os.makedirs(os.path.dirname(output_csv), exist_ok=True)

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "blur", "brightness", "noise", "best_pipeline"])

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset images folder not found: {dataset_path}")

    for file in os.listdir(dataset_path):

        if not file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        path = os.path.join(dataset_path, file)
        img = cv2.imread(path)

        if img is None:
            print(f"Skipping invalid image: {file}")
            continue

        blur, bright, noise = extract_features(img)
        pipeline = decide_pipeline(blur, bright, noise)

        writer.writerow([file, blur, bright, noise, pipeline])

print("labels.csv generated.")
