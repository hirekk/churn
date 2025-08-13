# 🚀 Telecom Customer Churn Prediction

A complete machine learning pipeline for predicting telecom customer churn using Airflow, MLflow, and Docker.

## 📋 Prerequisites

- **Docker** - [Install Docker](https://docs.docker.com/get-docker/)
- **Terraform** - [Install Terraform](https://developer.hashicorp.com/terraform/downloads)
- **Kaggle API credentials** - [Get your Kaggle API key](https://github.com/Kaggle/kaggle-api#api-credentials)

## 🚀 Quick Start

### 1. Setup and Deploy Infrastructure

```bash
# Clone and setup
git clone git@github.com:hirekk/churn.git
cd churn

# Configure Kaggle credentials
cat > terraform.tfvars << EOF
kaggle_username = "your_kaggle_username"
kaggle_key = "your_kaggle_api_key"
EOF

# Deploy infrastructure
terraform init
terraform apply -auto-approve
```

This starts:
- **Airflow** server on port 8081
- **MLflow** tracking server on port 5002
- **Local Docker registry** on port 5003

### 2. Get Airflow Admin Password

```bash
docker logs airflow-server | grep "Admin user" | tail -1
```

### 3. Access Services

- **Airflow UI**: http://localhost:8081 (admin / password from step 2)
- **MLflow UI**: http://localhost:5002

### 4. Train Model

1. **Open Airflow UI** → find `model_training` DAG
2. **Click "Play" button** to trigger training
3. **Monitor execution** in DAG view
4. **Note the Run ID** from the successful training run

The pipeline will:
- Download telecom customer churn dataset from Kaggle
- Apply feature engineering
- Train a Random Forest model with preprocessing pipeline
- Save the model to `models/random_forest/{run_id}`
- Log everything to MLflow

### 5. Deploy Prediction API

**Important**: Wait for training to complete and note the Run ID before deploying the API.

```bash
# Deploy API with specific trained model
make deploy RUN_ID="your_run_id_here"

# Check health
make check

# Test predictions
make test
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Kaggle Data  │    │   Airflow DAG   │    │   MLflow UI     │
│   (Download)   │───▶│   (Training)    │───▶│   (Tracking)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Local Data   │    │   Trained Model │    │   Model Files   │
│   (data/)      │    │   (models/)     │    │   (Deployment)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
churn/
├── airflow/                 # Airflow configuration
│   ├── dags/              # DAG definitions
│   └── logs/              # Airflow logs
├── data/                   # Data storage
│   ├── raw/               # Raw downloaded data
│   ├── interim/           # Intermediate processed data
│   └── processed/         # ML-ready features
├── models/                 # Trained models
│   └── random_forest/     # Model files by run ID
├── mlflow/                 # MLflow artifacts
├── src/                    # Source code
│   └── churn/             # Main package
└── main.tf                 # Terraform configuration
```

## 🚀 Deploy Prediction API

### Option 1: Using Makefile (Recommended)

```bash
# Deploy API with specific model
make deploy RUN_ID="your_run_id_here"

# Check health
make check

# Test predictions
make test

# Stop service
make down

# Clean up everything
make clean
```

### Option 2: Manual Docker Commands

```bash
# Build and run with specific model
docker build --build-arg RUN_ID="your_run_id" -f Dockerfile.api -t churn-api:latest .
docker run -d --name churn-api -p 8000:8000 churn-api:latest

# Test
curl http://localhost:8000/health
curl http://localhost:8000/docs
```

## 🧪 API Testing

### Health Check
```bash
curl http://localhost:8000/health
```

### Make Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "gender": "Male",
    "married": "Yes",
    "tenure_in_months": 24,
    "monthly_charge": 89.99
  }'
```

**Note**: This is a minimal example. The full API expects all customer fields. See the Makefile test target for a complete example.

### API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔧 Configuration

| Service | Port | Purpose |
|---------|------|---------|
| Airflow | 8081 | Web UI and API |
| MLflow | 5002 | Experiment tracking |
| Registry | 5003 | Docker image storage |
| API | 8000 | Prediction endpoint |

## 🚨 Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure ports 8081, 5002, 5003, 8000 are available
2. **Kaggle credentials**: Verify your API key is correct
3. **Docker not running**: Start Docker before running Terraform
4. **API deployment before training**: Ensure training completes and you have a valid RUN_ID

### Debug Commands

```bash
# Check container status
docker ps

# View logs
docker logs airflow-server
docker logs churn-api

# Check model files
ls -la models/random_forest/

# Restart services
terraform destroy -auto-approve
terraform apply -auto-approve
```

## 🔄 Model Updates

To deploy a new model version:

1. **Train new model** using Airflow
2. **Get the new Run ID** from MLflow or Airflow logs
3. **Redeploy API** with new model:
   ```bash
   make clean
   make deploy RUN_ID="new_run_id"
   ```

## 📚 Next Steps

- **Model Monitoring**: Set up performance monitoring
- **CI/CD Pipeline**: Automate model retraining
- **Production Deployment**: Scale to production environment

---

**Happy Churn Prediction! 🎯**
