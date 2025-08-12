"""FastAPI application for churn prediction model serving."""

import logging
import os

from fastapi import FastAPI
from fastapi import HTTPException
import mlflow.sklearn
import pandas as pd
from pydantic import BaseModel
from pydantic import Field
import uvicorn

import mlflow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="API for predicting telecom customer churn",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Model loading
MODEL_PATH = os.getenv("MODEL_PATH", "models/random_forest")
model = None


class CustomerData(BaseModel):
    """Customer data schema for prediction."""

    age: int = Field(..., ge=0, le=120, description="Customer age")
    gender: str = Field(..., description="Customer gender")
    married: str = Field(..., description="Marital status")
    number_of_dependents: int = Field(..., ge=0, description="Number of dependents")
    number_of_referrals: int = Field(..., ge=0, description="Number of referrals")
    tenure_in_months: int = Field(..., ge=0, description="Tenure in months")
    offer: str = Field(..., description="Offer type")
    phone_service: str = Field(..., description="Phone service")
    avg_monthly_long_distance_charges: float = Field(
        ..., ge=0, description="Average monthly long distance charges"
    )
    multiple_lines: str = Field(..., description="Multiple lines")
    internet_service: str = Field(..., description="Internet service")
    internet_type: str = Field(..., description="Internet type")
    avg_monthly_gb_download: float = Field(..., ge=0, description="Average monthly GB download")
    online_security: str = Field(..., description="Online security")
    online_backup: str = Field(..., description="Online backup")
    device_protection_plan: str = Field(..., description="Device protection plan")
    premium_tech_support: str = Field(..., description="Premium tech support")
    streaming_tv: str = Field(..., description="Streaming TV")
    streaming_movies: str = Field(..., description="Streaming movies")
    streaming_music: str = Field(..., description="Streaming music")
    unlimited_data: str = Field(..., description="Unlimited data")
    contract: str = Field(..., description="Contract type")
    paperless_billing: str = Field(..., description="Paperless billing")
    payment_method: str = Field(..., description="Payment method")
    monthly_charge: float = Field(..., ge=0, description="Monthly charge")
    total_charges: float = Field(..., ge=0, description="Total charges")
    total_refunds: float = Field(..., ge=0, description="Total refunds")
    total_extra_data_charges: float = Field(..., ge=0, description="Total extra data charges")
    total_long_distance_charges: float = Field(..., ge=0, description="Total long distance charges")
    total_revenue: float = Field(..., ge=0, description="Total revenue")
    latitude: float = Field(..., description="Latitude")
    longitude: float = Field(..., description="Longitude")


class PredictionResponse(BaseModel):
    """Prediction response schema."""

    customer_id: str
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn")
    churn_prediction: bool = Field(..., description="Churn prediction (True/False)")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")


def load_model() -> None:
    """Load the trained model from MLflow."""
    global model
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        model = mlflow.sklearn.load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.exception(f"Failed to load model: {e}")
        msg = f"Model loading failed: {e}"
        raise RuntimeError(msg)


def preprocess_input(customer_data: CustomerData) -> pd.DataFrame:
    """Preprocess customer data for model prediction."""
    # Convert to DataFrame
    df = pd.DataFrame([customer_data.dict()])

    # Apply the same preprocessing as during training
    from churn.data.feature_engineering import create_features

    # Create features (this should match your training pipeline)
    processed_df = create_features(df)

    # Remove target column if it exists
    if "target" in processed_df.columns:
        processed_df = processed_df.drop("target", axis=1)

    return processed_df


@app.on_event("startup")
async def startup_event() -> None:
    """Load model on startup."""
    load_model()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Churn Prediction API", "version": "1.0.0", "status": "healthy"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer_data: CustomerData, customer_id: str = "unknown"):
    """Predict customer churn probability."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Preprocess input data
        processed_data = preprocess_input(customer_data)

        # Make prediction
        churn_probability = model.predict_proba(processed_data)[0][1]
        churn_prediction = churn_probability > 0.5

        # Calculate confidence (distance from decision boundary)
        confidence = abs(churn_probability - 0.5) * 2

        logger.info(
            f"Prediction for customer {customer_id}: churn_prob={churn_probability:.3f}, prediction={churn_prediction}"
        )

        return PredictionResponse(
            customer_id=customer_id,
            churn_probability=churn_probability,
            churn_prediction=churn_prediction,
            confidence=confidence,
        )

    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e!s}")


@app.post("/predict_batch")
async def predict_batch(customers: list[CustomerData]):
    """Predict churn for multiple customers."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        results = []
        for i, customer_data in enumerate(customers):
            processed_data = preprocess_input(customer_data)
            churn_probability = model.predict_proba(processed_data)[0][1]
            churn_prediction = churn_probability > 0.5
            confidence = abs(churn_probability - 0.5) * 2

            results.append({
                "customer_id": f"customer_{i}",
                "churn_probability": churn_probability,
                "churn_prediction": churn_prediction,
                "confidence": confidence,
            })

        return {"predictions": results}

    except Exception as e:
        logger.exception(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e!s}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
