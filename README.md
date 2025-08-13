# Telecom Customer Churn Prediction

A complete machine learning pipeline for predicting telecom customer churn using Airflow, MLflow, and Docker.

## 📋 Prerequisites

- **Docker** - [Install Docker](https://docs.docker.com/get-docker/)
- **Terraform** - [Install Terraform](https://developer.hashicorp.com/terraform/downloads)
- **Kaggle API credentials** - [Get your Kaggle API key](https://github.com/Kaggle/kaggle-api#api-credentials)

## 🚀 Quick Start

### 1. Setup and Deploy Infrastructure

```bash
# Clone and setup
git clone https://github.com/hirekk/churn.git
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

> ⚠️ **Note**: The Airflow server may take a minute to become fully responsive while it initializes the database and creates the admin user. Wait until you see "Admin user" logs before proceeding.

```bash
docker logs airflow-server | grep "Admin user" | tail -1
```

### 3. Access Services

- **Airflow UI**: http://localhost:8081 (Username `admin` / password from step 2)
- **MLflow UI**: http://localhost:5002

### 4. Train Model

1. **Open Airflow UI** → find `model_training` DAG
2. **Click "Play" button** to trigger training
3. **Monitor execution** in DAG view
4. **Note the corresponding Run ID in the MLflow console** from the successful training run

The pipeline will:
- Download telecom customer churn dataset from Kaggle
- Apply feature engineering
- Train a Random Forest model with preprocessing pipeline
- Test and evaluate the model on a hold-out dataset.
- Save the model to `models/random_forest/{run_id}`
- Log everything to MLflow

### 5. Deploy Prediction API

**Important**: Wait for training to complete and note the Run ID before deploying the API.

```bash
# Deploy API with specific trained model
make deploy RUN_ID="your_run_id_here"

# Check service health
make check

# Test predictions
make test
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


## 🚨 Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure ports 8081, 5002, 5003, 8000 are available
2. **Kaggle credentials**: Verify your API key is correct
3. **Docker not running**: Start Docker before running Terraform
4. **API deployment before training**: Ensure training completes and you have a valid RUN_ID


## 🧭 Next Steps

- **Model Monitoring**: Set up service monitoring
- **CI/CD Pipeline**: Automate model retraining
- **Production Deployment**: Scale to production environment

---
