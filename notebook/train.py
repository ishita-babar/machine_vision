import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
labels_csv = os.path.join(PROJECT_ROOT, "dataset", "labels.csv")
model_dir = os.path.join(PROJECT_ROOT, "backend", "model")
model_path = os.path.join(model_dir, "model.pkl")

df = pd.read_csv(labels_csv)

if df.empty:
    raise ValueError(f"No rows in {labels_csv}. Run generate_labels.py first.")

X = df[["blur", "brightness", "noise"]]
y = df["best_pipeline"]

model = DecisionTreeClassifier()
model.fit(X, y)

os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, model_path)

print("Model trained successfully.")
