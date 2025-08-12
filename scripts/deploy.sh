#!/bin/bash

# Deploy script for Churn Prediction API
set -e

# Configuration
IMAGE_NAME="churn-prediction-api"
IMAGE_TAG="latest"
CONTAINER_NAME="churn-api"
REGISTRY="localhost:5000"
FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "🚀 Deploying Churn Prediction API..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if local registry is running
if ! curl -s http://localhost:5000/v2/ > /dev/null; then
    echo "⚠️  Local registry not running at localhost:5000"
    echo "   Starting local registry..."
    docker run -d -p 5000:5000 --name registry registry:2
    sleep 5
fi

# Build Docker image
echo "🔨 Building Docker image..."
docker build -f Dockerfile.api -t ${IMAGE_NAME}:${IMAGE_TAG} .

# Tag for local registry
echo "🏷️  Tagging image for local registry..."
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${FULL_IMAGE_NAME}

# Push to local registry
echo "📤 Pushing to local registry..."
docker push ${FULL_IMAGE_NAME}

# Stop and remove existing container if running
if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
    echo "🛑 Stopping existing container..."
    docker stop ${CONTAINER_NAME}
    docker rm ${CONTAINER_NAME}
fi

# Run new container
echo "🚀 Starting new container..."
docker run -d \
    --name ${CONTAINER_NAME} \
    -p 8000:8000 \
    -e MODEL_PATH=/app/models/random_forest \
    --restart unless-stopped \
    ${FULL_IMAGE_NAME}

# Wait for container to be ready
echo "⏳ Waiting for service to be ready..."
sleep 10

# Health check
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Service is healthy and running!"
    echo "🌐 API available at: http://localhost:8000"
    echo "📚 API docs at: http://localhost:8000/docs"
    echo "🔍 Health check: http://localhost:8000/health"
else
    echo "❌ Service health check failed"
    echo "📋 Container logs:"
    docker logs ${CONTAINER_NAME}
    exit 1
fi

echo "🎉 Deployment completed successfully!"
