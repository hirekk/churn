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

from churn.data import TARGET_VARIABLE
from churn.data.features import create_features
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Churn Prediction API",
    description="API for predicting telecom customer churn",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/random_forest")
model = None


class CustomerData(BaseModel):
    """Customer data schema for prediction."""

    customer_id: str = Field(..., alias="Customer ID", description="Customer ID")
    age: int = Field(..., alias="Age", ge=0, le=120, description="Customer age")
    gender: str = Field(..., alias="Gender", description="Customer gender")
    married: str = Field(..., alias="Married", description="Marital status")
    number_of_dependents: int = Field(
        ...,
        alias="Number of Dependents",
        ge=0,
        description="Number of dependents",
    )
    number_of_referrals: int = Field(
        ...,
        alias="Number of Referrals",
        ge=0,
        description="Number of referrals",
    )
    tenure_in_months: int = Field(
        ...,
        alias="Tenure in Months",
        ge=0,
        description="Tenure in months",
    )
    offer: str = Field(..., alias="Offer", description="Offer type")
    phone_service: str = Field(..., alias="Phone Service", description="Phone service")
    avg_monthly_long_distance_charges: float = Field(
        ...,
        alias="Avg Monthly Long Distance Charges",
        ge=0,
        description="Average monthly long distance charges",
    )
    multiple_lines: str = Field(..., alias="Multiple Lines", description="Multiple lines")
    internet_service: str = Field(..., alias="Internet Service", description="Internet service")
    internet_type: str = Field(..., alias="Internet Type", description="Internet type")
    avg_monthly_gb_download: float = Field(
        ...,
        alias="Avg Monthly GB Download",
        ge=0,
        description="Average monthly GB download",
    )
    online_security: str = Field(..., alias="Online Security", description="Online security")
    online_backup: str = Field(..., alias="Online Backup", description="Online backup")
    device_protection_plan: str = Field(
        ...,
        alias="Device Protection Plan",
        description="Device protection plan",
    )
    premium_tech_support: str = Field(
        ...,
        alias="Premium Tech Support",
        description="Premium tech support",
    )
    streaming_tv: str = Field(..., alias="Streaming TV", description="Streaming TV")
    streaming_movies: str = Field(..., alias="Streaming Movies", description="Streaming movies")
    streaming_music: str = Field(..., alias="Streaming Music", description="Streaming music")
    unlimited_data: str = Field(..., alias="Unlimited Data", description="Unlimited data")
    contract: str = Field(..., alias="Contract", description="Contract type")
    paperless_billing: str = Field(..., alias="Paperless Billing", description="Paperless billing")
    payment_method: str = Field(..., alias="Payment Method", description="Payment method")
    monthly_charge: float = Field(..., alias="Monthly Charge", ge=0, description="Monthly charge")
    total_charges: float = Field(..., alias="Total Charges", ge=0, description="Total charges")
    total_refunds: float = Field(..., alias="Total Refunds", ge=0, description="Total refunds")
    total_extra_data_charges: float = Field(
        ...,
        alias="Total Extra Data Charges",
        ge=0,
        description="Total extra data charges",
    )
    total_long_distance_charges: float = Field(
        ...,
        alias="Total Long Distance Charges",
        ge=0,
        description="Total long distance charges",
    )
    total_revenue: float = Field(..., alias="Total Revenue", ge=0, description="Total revenue")
    latitude: float = Field(..., alias="Latitude", description="Latitude")
    longitude: float = Field(..., alias="Longitude", description="Longitude")


class PredictionResponse(BaseModel):
    """Prediction response schema."""

    customer_id: str = Field(..., alias="Customer ID", description="Customer ID")
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn")
    churn_prediction: bool = Field(..., description="Churn prediction (True/False)")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")


def load_model() -> None:
    """Load the trained model from MLflow."""
    global model
    try:
        logger.info("Loading model from %s", MODEL_PATH)
        model = mlflow.sklearn.load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        msg = f"Model loading failed: {e}"
        raise RuntimeError(msg)


def preprocess_input(customer_data: CustomerData) -> pd.DataFrame:
    """Preprocess customer data for model prediction."""
    df = pd.DataFrame([customer_data.model_dump(by_alias=True)])

    processed_df = create_features(df)

    if TARGET_VARIABLE in processed_df.columns:
        processed_df = processed_df.drop(TARGET_VARIABLE, axis=1)

    # Debug logging to see what columns we have
    logger.info("API created features with columns: %s", list(processed_df.columns))
    logger.info("API feature count: %s", len(processed_df.columns))

    return processed_df


@app.on_event("startup")
def startup_event() -> None:
    """Load model on startup."""
    load_model()


@app.get("/")
def main():
    """Health check endpoint."""
    return {
        "message": "Churn Prediction API",
        "version": "1.0.0",
        "status": "healthy",
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer_data: CustomerData):
    """Predict customer churn probability."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Debug logging to see what the model expects
    if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
        preprocessor = model.named_steps["preprocessor"]
        if hasattr(preprocessor, "get_feature_names_out"):
            expected_features = preprocessor.get_feature_names_out()
            logger.info("Model expects features: %s", list(expected_features))
            logger.info("Model expected feature count: %s", len(expected_features))

    try:
        processed_data = preprocess_input(customer_data)
        logger.info(
            "API created features with columns: %s",
            processed_data.head(3).to_dict(orient="records"),
        )
        churn_probability = model.predict_proba(processed_data)[0][1]
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    churn_prediction = churn_probability > 0.5
    confidence = abs(churn_probability - 0.5) * 2
    logger.info(
        "Prediction for customer %s: churn_prob=%s, prediction=%s",
        customer_data.customer_id,
        churn_probability,
        churn_prediction,
    )

    return PredictionResponse(
        customer_id=customer_data.customer_id,
        churn_probability=churn_probability,
        churn_prediction=churn_prediction,
        confidence=confidence,
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
