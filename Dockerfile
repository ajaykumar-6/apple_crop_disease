# -------------------------------
# Render-compatible Docker setup
# -------------------------------

# Use Ubuntu + Python 3.11 base image
FROM python:3.11-slim

# Prevent interactive prompts & cache pip
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Create work directory
WORKDIR /app

# Copy dependency list
COPY requirements.txt .

# Install system libs TensorFlow often needs
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libhdf5-dev && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy rest of the project
COPY . .

# Expose Flask port
EXPOSE 10000

# Start the Flask app with Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]
