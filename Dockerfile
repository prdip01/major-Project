# Multi-Vendor Production Dockerfile
# ==================================
# Use the official Python slim runtime for a lightweight, secure container
FROM python:3.12-slim

# Set standard Python environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    FLASK_DEBUG=false

# Set workspace directory
WORKDIR /app

# Install critical system libraries required for OpenCV (cv2) and graphics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements list first to leverage Docker layer caching
COPY backend/requirements.txt /app/backend/requirements.txt

# Install python packages
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy application backend codebase and pretrained models
COPY backend/ /app/backend/
COPY models/ /app/models/

# Copy unified static frontend asset directory
COPY frontend/ /app/frontend/

# Expose default port (customizable via env var PORT)
EXPOSE 7860

# Start the application using a robust, concurrency-supported server (Gunicorn)
# Evaluates the dynamic $PORT environment variable assigned by hosting vendors
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --chdir backend app:app"]
