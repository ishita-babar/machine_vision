"""
Flask web UI for document restoration (uses document_restore.restore_pipeline).

Run from this directory:
  python app.py
Then open http://127.0.0.1:5000
"""

from __future__ import annotations

import os
from io import BytesIO

import cv2
import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_file

from document_restore import restore_pipeline

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(
    os.environ.get("MAX_UPLOAD_MB", "16")
) * 1024 * 1024

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _decode_image(data: bytes) -> np.ndarray | None:
    buf = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _encode_png(bgr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return encoded.tobytes()


def _allowed_file(filename: str) -> bool:
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return f".{ext}" in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/restore")
def restore():
    if "image" not in request.files:
        abort(400, description="No file field 'image'.")
    f = request.files["image"]
    if not f.filename or not _allowed_file(f.filename):
        abort(
            400,
            description="Invalid or missing file. Use PNG, JPEG, BMP, TIFF, or WebP.",
        )
    raw = f.read()
    if not raw:
        abort(400, description="Empty file.")

    bgr = _decode_image(raw)
    if bgr is None:
        abort(400, description="Could not decode image.")

    shadow_removal = request.form.get("shadow_removal", "1") == "1"
    deskew = request.form.get("deskew", "1") == "1"
    perspective = request.form.get("perspective", "0") == "1"

    try:
        out_bgr, _meta = restore_pipeline(
            bgr,
            shadow_removal=shadow_removal,
            deskew=deskew,
            perspective=perspective,
            debug=False,
        )
        png_bytes = _encode_png(out_bgr)
    except Exception as e:  # noqa: BLE001 — surface pipeline errors to client
        abort(500, description=str(e))

    return send_file(
        BytesIO(png_bytes),
        mimetype="image/png",
        as_attachment=False,
        download_name="restored.png",
    )


@app.errorhandler(400)
@app.errorhandler(413)
@app.errorhandler(500)
def http_error(err):
    if getattr(err, "description", None):
        msg = err.description
    else:
        msg = getattr(err, "name", None) or "Error"
    code = getattr(err, "code", 500) or 500
    wants_json = (
        request.path == "/restore"
        and request.method == "POST"
    ) or "application/json" in (request.headers.get("Accept") or "")
    if wants_json:
        return jsonify({"error": msg}), code
    return msg, code


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=True)
