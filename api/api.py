# api/api.py

# --- 1. Imports ---
# Third-party libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import joblib
import uvicorn

# Your custom local modules from the 'src' directory
# Make sure these files exist and the functions are defined correctly.
from src.parser import parse_line
from src.template_extractor import normalize_msg


# --- 2. Pydantic Models for Data Validation ---
# Defines the structure for incoming API requests
class LogRequest(BaseModel):
    lines: List[str]

# Defines the structure for the /parse endpoint's response
class ParseResponse(BaseModel):
    timestamp: Optional[str]
    level: Optional[str]
    service: Optional[str]
    msg: str
    raw: str

# Defines the structure for the /predict endpoint's response
class PredictionResponse(BaseModel):
    raw: str
    anomaly_score: float
    is_anomaly: bool


# --- 3. FastAPI App Initialization & Model Loading ---
app = FastAPI(
    title="Log Anomaly Detection API",
    description="An API to parse logs and predict anomalies using a pre-trained model.",
    version="1.0.0"
)

# Load the trained machine learning models when the API starts
# Using a try-except block handles the case where models don't exist yet.
try:
    rf_model = joblib.load('models/rf_anomaly_detector.joblib')
    vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
    print("INFO: Machine learning models loaded successfully.")
except FileNotFoundError:
    rf_model = None
    vectorizer = None
    print("WARNING: Model files not found. The /predict endpoint will be disabled.")


# --- 4. API Endpoints ---
@app.get("/health")
def health_check():
    """
    Health check endpoint to verify API status and model loading.
    """
    model_status = "Models loaded successfully."
    if rf_model is None or vectorizer is None:
        model_status = "Models not found, /predict endpoint is disabled."

    return {
        "status": "ok",
        "model_status": model_status
    }

@app.post("/parse", response_model=List[ParseResponse])
def parse_log_lines(request: LogRequest):
    """
    Parses a list of raw log lines and returns structured data.
    """
    parsed_logs = [parse_line(line) for line in request.lines]
    return parsed_logs

@app.post("/predict", response_model=List[PredictionResponse])
def predict_anomalies(request: LogRequest):
    """
    Parses log lines and predicts if they are anomalies.
    """
    # First, check if the models were loaded successfully on startup
    if rf_model is None or vectorizer is None:
        raise HTTPException(
            status_code=503, # Service Unavailable
            detail="Models are not available. Please train the models first."
        )

    # Process the incoming log lines
    parsed_logs = [parse_line(line) for line in request.lines]
    normalized_msgs = [normalize_msg(p['msg']) for p in parsed_logs]

    # Use the trained vectorizer and model to make predictions
    X_transformed = vectorizer.transform(normalized_msgs)
    anomaly_probabilities = rf_model.predict_proba(X_transformed)[:, 1]

    # Format the response
    response_data = []
    for p, prob in zip(parsed_logs, anomaly_probabilities):
        response_data.append({
            "raw": p['raw'],
            "anomaly_score": float(prob),
            "is_anomaly": bool(prob > 0.5) # Using 0.5 as the threshold
        })

    return response_data


# --- 5. Main Block for Direct Execution ---
# This allows you to run the API directly with "python api/api.py"
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)