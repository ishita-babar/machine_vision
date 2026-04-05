from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import os
import joblib

from utils.feature_extraction import extract_features
from processing import process_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "temp")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output", "steps")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

app = Flask(__name__)
CORS(app)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Missing model at {MODEL_PATH}. Run notebook/train.py first."
    )

model = joblib.load(MODEL_PATH)


def imread_unicode(path):
    """Read image on Windows paths with non-ASCII characters."""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


@app.route("/")
def home():
    return "API Running"


@app.route("/process", methods=["POST"])
def process():
    if "image" not in request.files:
        return jsonify({"error": "No image file (field name must be 'image')."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    filepath = os.path.join(UPLOAD_FOLDER, os.path.basename(file.filename))
    file.save(filepath)

    image = imread_unicode(filepath)
    if image is None:
        return jsonify({"error": "Could not read image."}), 400

    features = extract_features(image)
    decision = str(model.predict([features])[0])

    steps = process_pipeline(image, decision)

    output_paths = {}

    for step_name, img in steps.items():
        filename = f"{step_name}.png"
        save_path = os.path.join(OUTPUT_FOLDER, filename)
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            return jsonify({"error": f"Failed to encode step: {step_name}"}), 500
        buf.tofile(save_path)
        output_paths[step_name] = f"/output/{filename}"

    return jsonify({
        "decision": decision,
        "steps": output_paths,
        "step_order": list(steps.keys()),
    })


@app.route("/output/<filename>")
def serve_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
