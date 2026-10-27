# Makefile for Log Anomaly Detection & Forensic Analysis Pipeline

# Variables
PYTHON = python
PIP = pip
UVicorn = uvicorn
STREAMLIT = streamlit
PYTEST = pytest
DOCKER_COMPOSE = docker-compose

.PHONY: install train test run-api run-app docker-build docker-up clean help

help:
	@echo "Available commands:"
	@echo "  make install        - Install python library dependencies"
	@echo "  make train          - Run offline phase (Preprocess logs + Train RF & LSTM models)"
	@echo "  make test           - Run pytest unit tests"
	@echo "  make run-api        - Start the FastAPI backend server locally"
	@echo "  make run-app        - Start the Streamlit visualization dashboard locally"
	@echo "  make docker-build   - Build container images using Docker Compose"
	@echo "  make docker-up      - Build and spin up the complete container architecture"
	@echo "  make clean          - Remove Python compilation and cache folders"

install:
	$(PIP) install -r requirements.txt

train:
	$(PYTHON) src/dataset_builder.py
	$(PYTHON) src/train_rf_explain.py
	$(PYTHON) src/train_lstm.py

test:
	$(PYTEST) tests/

run-api:
	$(UVicorn) api.api:app --reload --host 0.0.0.0 --port 8000

run-app:
	$(STREAMLIT) run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

docker-build:
	$(DOCKER_COMPOSE) build

docker-up:
	$(DOCKER_COMPOSE) up

clean:
	rm -rf __pycache__ src/__pycache__ api/__pycache__ tests/__pycache__
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
