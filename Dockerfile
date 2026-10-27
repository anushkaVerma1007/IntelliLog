# Dockerfile

# --- Stage 1: Build dependencies ---
FROM python:3.9-slim AS builder

WORKDIR /app

# Install system compilation dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


# --- Stage 2: Final lightweight execution image ---
FROM python:3.9-slim

WORKDIR /app

# Copy installed python dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Copy project source and data
COPY . .

# Expose Streamlit (8501) and FastAPI (8000)
EXPOSE 8000
EXPOSE 8501

# Default command runs the FastAPI server
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]
