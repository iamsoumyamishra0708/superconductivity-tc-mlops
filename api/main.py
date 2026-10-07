"""
FastAPI Model Deployment Module (Step 3).

This module exposes the trained XGBoost model as a REST API.
It includes endpoints for health checks, retrieving expected features, 
and making predictions on new material data.

Team Instructions:
- Pre-requisite: Ensure `src/train.py` has been run successfully so that 
  `models/model.joblib` and `models/features.json` exist before starting the server.
- Local Development: Execute `$ uvicorn api.main:app --reload` from the project root.
- CORS Policy: CORS is currently configured with `allow_origins=["*"]` for local frontend 
  development. IMPORTANT: Before deploying to production, this must be restricted to 
  the specific frontend domain.
- Testing: Do not push changes to this file without running the automated test suite 
  (`$ pytest tests/test_api.py`) to verify endpoint stability.
"""

# Importing standard libraries for file path operations and JSON handling
import os
import json

# Importing pandas to convert incoming JSON data into a DataFrame for the model
import pandas as pd

# Importing joblib for loading the serialized (saved) machine learning model
import joblib

# Importing asynccontextmanager to handle the startup/shutdown lifecycle of the FastAPI app
from contextlib import asynccontextmanager

# Importing FastAPI for creating the web server, and HTTPException for error handling
from fastapi import FastAPI, HTTPException

# Importing CORSMiddleware so our HTML frontend can talk to this API
from fastapi.middleware.cors import CORSMiddleware

# Importing BaseModel from pydantic to define the strict data schema for incoming API requests
from pydantic import BaseModel

# Importing Dict from typing to specify the exact format of our features payload
from typing import Dict

# Define absolute paths to locate the saved model and features files reliably
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'model.joblib')
FEATURES_PATH = os.path.join(MODELS_DIR, 'features.json')

# Global variables to store the loaded model and feature list in memory
model = None
expected_features = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager. 
    This runs once when the server starts up, loading the model into memory 
    so it doesn't have to be reloaded from disk for every single prediction request.
    """
    global model, expected_features
    
    # Check if the required artifact files exist before attempting to load them
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        model = joblib.load(MODEL_PATH)
        with open(FEATURES_PATH, "r") as f:
            expected_features = json.load(f)
        print("✅ Model and features loaded successfully!")
    else:
        print("❌ Error: Model artifacts not found. Please run src/train.py first.")
        
    yield  # Yield control back to FastAPI

# Initialize the FastAPI application with the defined lifespan
app = FastAPI(title="Superconductivity Tc Predictor", version="1.0", lifespan=lifespan)

# CORS setup to allow the HTML file to securely connect to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all websites/local HTML files to connect
    allow_credentials=True,
    allow_methods=["*"], # Allows GET, POST, etc.
    allow_headers=["*"], # Allows all headers
)

class PredictRequest(BaseModel):
    """
    Schema for the prediction request body.
    Expects a JSON dictionary mapping feature names (strings) to their numeric values (floats).
    """
    features: Dict[str, float]

@app.get("/")
def read_root():
    """
    Root endpoint to welcome users and confirm the API is live.
    """
    return {
        "message": "Welcome to the Superconductivity Tc Predictor API!",
        "documentation": "/docs",
        "endpoints": ["/health", "/features", "/predict"]
    }

@app.get("/health")
def health_check():
    """
    Health check endpoint to verify that the server is running 
    and the ML model is successfully loaded into memory.
    """
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/features")
def get_features():
    """
    Endpoint to retrieve the list of features expected by the model.
    Returns the total count and the full list of feature names.
    """
    if not expected_features:
        raise HTTPException(status_code=500, detail="Features not loaded.")
    
    return {
        "feature_count": len(expected_features), 
        "expected_features": expected_features
    }

@app.post("/predict")
def predict(request: PredictRequest):
    """
    Core prediction endpoint.
    Takes 81 material features via POST request and returns the predicted critical temperature (Tc).
    """
    if not model or not expected_features:
        raise HTTPException(status_code=500, detail="Model is not initialized.")
    
    # Validation: Check if any expected feature is missing in the request payload
    missing_features = [f for f in expected_features if f not in request.features]
    if missing_features:
        raise HTTPException(
            status_code=400, 
            detail=f"Missing {len(missing_features)} features. Example missing: {missing_features[:5]}"
        )
    
    # Prepare data for prediction (maintaining the exact column order expected by the model)
    input_data = {feat: [request.features[feat]] for feat in expected_features}
    df = pd.DataFrame(input_data)
    
    # Run the machine learning model prediction
    prediction = model.predict(df)
    
    # Return the predicted value as a JSON response
    return {"predicted_critical_temp_K": float(prediction[0])}