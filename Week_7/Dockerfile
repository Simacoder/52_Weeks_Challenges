FROM python:3.8-slim

WORKDIR /app

# Install system dependencies first
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

# Upgrade pip before installing dependencies
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "src/train.py"]
