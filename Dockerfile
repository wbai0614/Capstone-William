# Use a small, secure base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (optional, uncomment if you need them)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#  && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose the port Gunicorn will bind to
EXPOSE 8080

# Start Gunicorn HTTP server
# - 2 workers, thread worker class, 60s timeout, bind to all interfaces
CMD ["gunicorn", "-w", "2", "-k", "gthread", "-t", "60", "--bind", "0.0.0.0:8080", "app:app"]
