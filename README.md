# 🚀 Telecom Customer Churn Prediction

A complete machine learning pipeline for predicting telecom customer churn using Airflow, MLflow, and Docker.

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Docker** - [Install Docker](https://docs.docker.com/get-docker/)
- **Terraform** - [Install Terraform](https://developer.hashicorp.com/terraform/downloads)
- **Kaggle API credentials** - [Get your Kaggle API key](https://github.com/Kaggle/kaggle-api#api-credentials)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone git@github.com:hirekk/churn.git
cd churn
```

### 2. Configure Kaggle Credentials

Create a `terraform.tfvars` file with your Kaggle credentials:

```bash
# Create terraform.tfvars
cat > terraform.tfvars << EOF
kaggle_username = "your_kaggle_username"
kaggle_key = "your_kaggle_api_key"
EOF
```

### 3. Deploy Infrastructure

```bash
# Initialize Terraform
terraform init

# Deploy all services
terraform apply -auto-approve
```

This will start:
- **Airflow** server on port 8081
- **MLflow** tracking server on port 5002
- **Local Docker registry** on port 5003

### 4. Get Airflow Admin Password

```bash
# Extract admin password from container logs
docker logs airflow-server | grep "Admin user" | tail -1
```

### 5. Access the Services

- **Airflow UI**: http://localhost:8081
  - Username: `admin`
  - Password: (from step 4)

- **MLflow UI**: http://localhost:5002

### 6. Run the Training Pipeline

1. **Open Airflow UI** in your browser
2. **Go to DAGs** → find `model_training`
3. **Click "Play" button** to trigger the pipeline
4. **Monitor execution** in the DAG view

The pipeline will:
- Download telecom customer churn dataset from Kaggle
- Apply feature engineering
- Train a Random Forest model
- Save the model to `models/random_forest/`
- Log everything to MLflow

### 7. View Results

- **Check MLflow UI** for experiment tracking and model artifacts
- **Verify model files** in `models/random_forest/` directory
- **Review training logs** in Airflow task logs

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
│   ├── logs/              # Airflow logs
│   └── airflow.db         # Airflow database
├── data/                   # Data storage
│   ├── raw/               # Raw downloaded data
│   ├── interim/           # Intermediate processed data
│   └── processed/         # ML-ready features
├── models/                 # Trained models
│   └── random_forest/     # Random Forest model files
├── mlflow/                 # MLflow artifacts
│   ├── mlruns/            # Experiment runs
│   └── mlflow.db          # MLflow database
├── src/                    # Source code
│   └── churn/             # Main package
├── main.tf                 # Terraform configuration
└── README.md               # This file
```

## 🔧 Configuration

### Environment Variables

| Service | Port | Purpose |
|---------|------|---------|
| Airflow | 8081 | Web UI and API |
| MLflow | 5002 | Experiment tracking |
| Registry | 5003 | Docker image storage |

### Volume Mounts

| Container Path | Host Path | Purpose |
|----------------|-----------|---------|
| `/opt/airflow/dags` | `./airflow/dags` | DAG definitions |
| `/opt/airflow/data` | `./data` | Data storage |
| `/opt/airflow/models` | `./models` | Model storage |
| `/mlflow/mlruns` | `./mlflow/mlruns` | MLflow artifacts |

## 🚨 Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure ports 8081, 5002, 5003 are available
2. **Kaggle credentials**: Verify your API key is correct
3. **Docker not running**: Start Docker before running Terraform
4. **Permission errors**: Ensure Docker has access to your project directory

### Debug Commands

```bash
# Check container status
docker ps

# View container logs
docker logs airflow-server
docker logs mlflow-server

# Restart services
terraform destroy -auto-approve
terraform apply -auto-approve
```

## 📚 Next Steps

- **API Deployment**: Deploy the prediction API (coming soon)
- **Model Monitoring**: Set up model performance monitoring
- **CI/CD Pipeline**: Automate model retraining
- **Production Deployment**: Scale to production environment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Happy Churn Prediction! 🎯**
