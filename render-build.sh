#!/usr/bin/env bash
# exit on error
set -o errexit

echo "--- Installing System Dependencies for OCR & PDF Parsing ---"
apt-get update && apt-get install -y tesseract-ocr libtesseract-dev poppler-utils ffmpeg libsm6 libxext6

echo "--- Installing Python Dependencies ---"
pip install --upgrade pip
pip install -r requirements.txt

echo "--- Render Build Completed Successfully ---"
