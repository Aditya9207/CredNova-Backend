FROM python:3.10-slim

# Install Tesseract, Poppler, and OpenCV runtime system libraries
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose Render default port
EXPOSE 10000

# Start Uvicorn production server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
