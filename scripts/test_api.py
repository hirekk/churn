#!/usr/bin/env python3
"""Test script for Churn Prediction API."""

import requests


def test_health_check(base_url: str = "http://localhost:8000") -> bool:
    """Test health check endpoint."""
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            response.json()
            return True
        return False
    except Exception:
        return False


def test_prediction(base_url: str = "http://localhost:8000") -> bool:
    """Test prediction endpoint with sample data."""
    sample_customer = {
        "age": 45,
        "gender": "Male",
        "married": "Yes",
        "number_of_dependents": 2,
        "number_of_referrals": 1,
        "tenure_in_months": 24,
        "offer": "Offer A",
        "phone_service": "Yes",
        "avg_monthly_long_distance_charges": 15.50,
        "multiple_lines": "Yes",
        "internet_service": "Yes",
        "internet_type": "Fiber Optic",
        "avg_monthly_gb_download": 50.0,
        "online_security": "Yes",
        "online_backup": "No",
        "device_protection_plan": "Yes",
        "premium_tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "streaming_music": "No",
        "unlimited_data": "Yes",
        "contract": "Two Year",
        "paperless_billing": "Yes",
        "payment_method": "Credit Card",
        "monthly_charge": 89.99,
        "total_charges": 2159.76,
        "total_refunds": 0.0,
        "total_extra_data_charges": 25.00,
        "total_long_distance_charges": 372.00,
        "total_revenue": 2159.76,
        "latitude": 37.7749,
        "longitude": -122.4194,
    }

    try:
        response = requests.post(
            f"{base_url}/predict", json=sample_customer, params={"customer_id": "test_customer_001"}
        )

        if response.status_code == 200:
            response.json()
            return True
        return False
    except Exception:
        return False


def test_batch_prediction(base_url: str = "http://localhost:8000") -> bool:
    """Test batch prediction endpoint."""
    sample_customers = [
        {
            "age": 35,
            "gender": "Female",
            "married": "No",
            "number_of_dependents": 0,
            "number_of_referrals": 0,
            "tenure_in_months": 12,
            "offer": "Offer B",
            "phone_service": "Yes",
            "avg_monthly_long_distance_charges": 0.0,
            "multiple_lines": "No",
            "internet_service": "No",
            "internet_type": "None",
            "avg_monthly_gb_download": 0.0,
            "online_security": "No",
            "online_backup": "No",
            "device_protection_plan": "No",
            "premium_tech_support": "No",
            "streaming_tv": "No",
            "streaming_movies": "No",
            "streaming_music": "No",
            "unlimited_data": "No",
            "contract": "Month-to-Month",
            "paperless_billing": "No",
            "payment_method": "Mailed Check",
            "monthly_charge": 29.99,
            "total_charges": 359.88,
            "total_refunds": 0.0,
            "total_extra_data_charges": 0.0,
            "total_long_distance_charges": 0.0,
            "total_revenue": 359.88,
            "latitude": 34.0522,
            "longitude": -118.2437,
        }
    ]

    try:
        response = requests.post(f"{base_url}/predict_batch", json=sample_customers)

        if response.status_code == 200:
            response.json()
            return True
        return False
    except Exception:
        return False


def main() -> int:
    """Run all tests."""
    base_url = "http://localhost:8000"

    tests = [
        ("Health Check", test_health_check),
        ("Single Prediction", test_prediction),
        ("Batch Prediction", test_batch_prediction),
    ]

    passed = 0
    total = len(tests)

    for _test_name, test_func in tests:
        if test_func(base_url):
            passed += 1

    if passed == total:
        pass
    else:
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
