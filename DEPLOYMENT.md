# 🚀 Churn Prediction API Deployment Guide

This guide covers the complete deployment of the Churn Prediction API using Docker and local registry.

## 📋 Prerequisites

- Docker installed and running
- Python 3.12+ (for local development)
- MLflow tracking server running (for model export)
- Trained model available in MLflow

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MLflow Model │    │  Docker Image   │    │  FastAPI App   │
│   (Training)   │───▶│  (Container)    │───▶│  (/predict)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 Setup Steps

### 1. Export Trained Model

First, export your best model from MLflow:

```bash
# Get the run ID from your MLflow UI or training logs
MLFLOW_RUN_ID="your_run_id_here"

# Export the model
python scripts/export_model.py $MLFLOW_RUN_ID
```

This creates the `models/random_forest/` directory with your trained model.

### 2. Build and Deploy

Use the automated deployment script:

```bash
# Make script executable
chmod +x scripts/deploy.sh

# Run deployment
./scripts/deploy.sh
```

The script will:
- Build the Docker image
- Start a local registry (if not running)
- Push the image to the registry
- Run the container
- Perform health checks

### 3. Manual Deployment (Alternative)

If you prefer manual steps:

```bash
# Build image
docker build -f Dockerfile.api -t churn-prediction-api:latest .

# Start local registry
docker run -d -p 5000:5000 --name registry registry:2

# Tag and push to registry
docker tag churn-prediction-api:latest localhost:5000/churn-prediction-api:latest
docker push localhost:5000/churn-prediction-api:latest

# Run container
docker run -d \
    --name churn-api \
    -p 8000:8000 \
    -e MODEL_PATH=/app/models/random_forest \
    localhost:5000/churn-prediction-api:latest
```

## 🧪 Testing the API

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict?customer_id=test001" \
  -H "Content-Type: application/json" \
  -d @scripts/sample_customer.json
```

### Run Full Test Suite
```bash
python scripts/test_api.py
```

## 📚 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with API info |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API documentation |
| `/predict` | POST | Single customer prediction |
| `/predict_batch` | POST | Batch customer predictions |

## 🔍 API Documentation

Once deployed, visit:
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 🐳 Docker Commands

### Container Management
```bash
# View running containers
docker ps

# View logs
docker logs churn-api

# Stop container
docker stop churn-api

# Remove container
docker rm churn-api

# Restart container
docker restart churn-api
```

### Image Management
```bash
# List images
docker images

# Remove image
docker rmi churn-prediction-api:latest

# View image history
docker history churn-prediction-api:latest
```

### Registry Management
```bash
# View registry contents
curl http://localhost:5000/v2/_catalog

# View image tags
curl http://localhost:5000/v2/churn-prediction-api/tags/list
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/app/models/random_forest` | Path to model artifacts |
| `PYTHONPATH` | `/app` | Python path |
| `PYTHONUNBUFFERED` | `1` | Unbuffered Python output |

### Port Configuration

- **API**: 8000 (configurable in Dockerfile)
- **Registry**: 5000 (configurable in deploy script)

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Failed**
   - Check if model was exported correctly
   - Verify MODEL_PATH environment variable
   - Check container logs: `docker logs churn-api`

2. **Port Already in Use**
   - Stop existing container: `docker stop churn-api`
   - Check what's using port 8000: `lsof -i :8000`

3. **Registry Connection Failed**
   - Start registry: `docker run -d -p 5000:5000 --name registry registry:2`
   - Wait for registry to be ready

4. **Health Check Fails**
   - Check container status: `docker ps -a`
   - View logs: `docker logs churn-api`
   - Verify model files exist in container

### Debug Commands

```bash
# Enter container shell
docker exec -it churn-api /bin/bash

# Check model directory
docker exec churn-api ls -la /app/models/

# Test model loading
docker exec churn-api python -c "import mlflow; print('MLflow OK')"
```

## 📊 Monitoring

### Health Metrics
- Container health status
- Model loading status
- API response times
- Error rates

### Logs
```bash
# Follow logs in real-time
docker logs -f churn-api

# View last 100 lines
docker logs --tail 100 churn-api
```

## 🔄 Updates and Rollbacks

### Update Model
```bash
# Export new model
python scripts/export_model.py NEW_RUN_ID

# Rebuild and redeploy
./scripts/deploy.sh
```

### Rollback
```bash
# Stop current container
docker stop churn-api

# Run previous version
docker run -d --name churn-api -p 8000:8000 \
  localhost:5000/churn-prediction-api:previous_tag
```

## 🎯 Production Considerations

For production deployment, consider:

1. **Security**
   - HTTPS/TLS encryption
   - Authentication/Authorization
   - Rate limiting

2. **Scalability**
   - Load balancing
   - Horizontal scaling
   - Resource limits

3. **Monitoring**
   - Prometheus metrics
   - Grafana dashboards
   - Alerting

4. **CI/CD**
   - Automated testing
   - Blue-green deployments
   - Rollback strategies

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review container logs
3. Verify model export was successful
4. Ensure all prerequisites are met

---

**Happy Deploying! 🚀**
