"""
Model Training and Tracking Module (Step 2 - MLflow & SHAP).

This module trains an XGBoost regressor, logs parameters, metrics, and models 
using MLflow, generates SHAP explainability plots, and saves the final model 
artifacts for FastAPI deployment.
"""

import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import shap
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from data import get_train_test_splits

# Define paths relative to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

def train_and_log_model():
    """
    Trains the XGBoost model, logs metrics to MLflow, generates SHAP plots, 
    and saves deployment artifacts.
    """
    # Ensure output directories exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Loading data...")
    X_train, X_test, y_train, y_test = get_train_test_splits(use_grouped_split=True)
    
    # Step 1: Set up MLflow experiment
    mlflow.set_experiment("Superconductivity_Prediction")
    
    with mlflow.start_run():
        print("Training XGBoost model...")
        params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "random_state": 42
        }
        # Log parameters to MLflow
        mlflow.log_params(params)
        
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        print("Evaluating model...")
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        # Log metrics to MLflow
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        print(f"Results -> RMSE: {rmse:.2f} | R2: {r2:.2f}")
        
        # Step 2: Save model and features for FastAPI deployment
        print("Saving model artifacts...")
        joblib.dump(model, os.path.join(MODELS_DIR, "model.joblib"))
        with open(os.path.join(MODELS_DIR, "features.json"), "w") as f:
            json.dump(list(X_train.columns), f)
        
        # Log model object to MLflow
        mlflow.xgboost.log_model(model, "xgboost-model")

        # Step 3: Generate SHAP explainability plots
        print("Generating SHAP plots (this might take a few seconds)...")
        explainer = shap.TreeExplainer(model)
        # Using a subset (e.g., 500 samples) of test data to speed up SHAP calculation
        X_test_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)
        shap_values = explainer.shap_values(X_test_sample)
        
        # Plot 1: Summary Plot
        plt.figure()
        shap.summary_plot(shap_values, X_test_sample, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, "shap_summary.png"))
        plt.close()
        
        # Plot 2: Bar Plot (Feature Importance)
        plt.figure()
        shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, "shap_importance.png"))
        plt.close()

        print("Training pipeline completed successfully! Artifacts saved.")

if __name__ == "__main__":
    train_and_log_model()