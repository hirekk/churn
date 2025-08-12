# Churn Prediction Project

This project demonstrates a complete ML pipeline with MLflow for experiment tracking and a local Docker registry for model versioning.

## Prerequisites

- Docker Desktop installed and running
- Terraform installed
- Ports 5002 and 5003 available on your machine

## Quick Start

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd churn
   ```

2. **Set up Kaggle credentials**
   Create a `terraform.tfvars` file at project root with the following contents:
   ```terraform
   # Kaggle credentials for dataset download
   kaggle_username = "hirekk"
   kaggle_key      = "91df20de33eb8fc15f8492a3a2ed8cde"
   ```

3. **Create required directories**
   ```bash
   mkdir -p mlflow/runs
   touch mlflow/mlflow.db
   ```

3. **Build custom Airflow image**
   ```bash
   # Build the custom Airflow image with project dependencies
   docker build -f Dockerfile.airflow -t churn-airflow:latest .
   ```

4. **Deploy infrastructure**
   ```bash
   terraform init
   terraform apply
   ```

4. **Access services**
   - **MLflow UI**: http://localhost:5002
   - **Docker Registry**: http://localhost:5003
   - **Airflow UI**: http://localhost:8081

5. **Airflow credentials**

   - **Username**: `admin`
   - **Password**: `MNTR38XczcNUC5tt`

## What's Included

### MLflow Server (Port 5002)
- Experiment tracking and model management
- SQLite backend for simplicity
- Custom runs directory for artifacts
- Accessible at http://localhost:5002

### Docker Registry (Port 5003)
- Local image storage and versioning
- HTTP access (no TLS for local development)
- Delete enabled for development
- Accessible at http://localhost:5003

### Airflow (Port 8081)
- Workflow orchestration and scheduling
- Standalone mode with automatic admin user creation
- SQLite backend for simplicity
- Accessible at http://localhost:8081

## Updating Dependencies

When you change project dependencies (in `pyproject.toml` or `requirements.txt`):

1. **Rebuild the custom Airflow image**
   ```bash
   docker build -f Dockerfile.airflow -t churn-airflow:latest .
   ```

2. **Redeploy infrastructure**
   ```bash
   terraform apply
   ```

**Note**: The custom Airflow image includes all project dependencies, so you must rebuild it when dependencies change.

## Usage Examples

### Using the Docker Registry

```bash
# Build and tag your ML model
docker build -t churn-model:v1.0 .

# Push to local registry
docker tag churn-model:v1.0 localhost:5003/churn-model:v1.0
docker push localhost:5003/churn-model:v1.0

# Pull from local registry
docker pull localhost:5003/churn-model:v1.0

# List images in registry
curl http://localhost:5003/v2/_catalog
```

### Using MLflow

```bash
# Access the web UI
open http://localhost:5002

# Or use the Python API
import mlflow
mlflow.set_tracking_uri("http://localhost:5002")
```

## Project Structure

```
churn/
├── main.tf                 # Terraform infrastructure
├── mlflow/                 # MLflow data directory
│   ├── runs/              # Experiment runs and artifacts
│   └── mlflow.db          # SQLite database
├── dags/                   # Airflow DAGs (if using)
├── data/                   # Your dataset files
└── src/                    # Source code
```

## Troubleshooting

### Port Conflicts
- **Port 5000**: Reserved by Apple AirPlay on macOS
- **Port 5001**: Free (unused)
- **Port 5002**: MLflow (safe to use)
- **Port 5003**: Docker Registry (safe to use)
- **Port 8081**: Airflow (safe to use)

### Common Issues
1. **"Connection refused"**: Make sure Docker is running
2. **"Port already in use"**: Check if ports 5002/5003 are free
3. **"Permission denied"**: Ensure Docker has access to your directories

### Reset Everything
```bash
# Stop and remove containers
terraform destroy

# Clean up data (optional)
rm -rf mlflow/runs/* mlflow/mlflow.db
```

## Development Workflow

1. **Train models** and log experiments with MLflow
2. **Package models** as Docker images
3. **Push images** to local registry
4. **Track versions** in MLflow experiments
5. **Deploy models** from registry when ready

## Sharing with Team

1. **Share the entire repository** (including `main.tf`)
2. **Team members run** `terraform apply` locally
3. **Everyone gets** the same infrastructure setup
4. **Use consistent image naming** conventions
5. **Document model versions** in MLflow

## Next Steps

- Add your ML training code to `src/`
- Create Dockerfiles for your models
- Set up CI/CD pipelines
- Add monitoring and logging
- Scale to production infrastructure
