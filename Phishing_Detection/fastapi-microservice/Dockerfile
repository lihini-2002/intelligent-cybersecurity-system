# ------------ Base Image ------------
# Use a minimal base image with Python pre-installed
FROM python:3.10-slim as base

# ------------ Metadata ------------
LABEL maintainer="gayathriharshila22@gmail.com"
LABEL description="FastAPI microservice for phishing detection (SMS & URL classifiers)"

# ------------ Security: Unprivileged user (optional) ------------
# Create non-root user (helps mitigate container escape risks)
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# ------------ Working Directory ------------
# Create and set the app working directory inside container
WORKDIR /app

# ------------ Copy Dependency Files ------------
# Copy only what’s needed for dependency installation first (to leverage Docker layer caching)
COPY --chown=appuser:appuser requirements.txt .

# ------------ Install Dependencies ------------
# Install packages safely and cleanly
RUN pip install --no-cache-dir -r requirements.txt

# ------------ Copy Application Code ------------
COPY --chown=appuser:appuser main.py ./
COPY --chown=appuser:appuser models ./models

# ------------ Expose Port ------------
# FastAPI runs on port 8000 by default inside container
EXPOSE 8000

# ------------ Default Command ------------
# Use uvicorn to serve the app with auto-reload turned OFF (for production)
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

