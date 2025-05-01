# Reddit Sentiment Analysis App

This is a simple FastAPI application that provides sentiment analysis for Reddit comments. The application is containerized and can be deployed both locally using Minikube and on Google Cloud Platform using GKE.

## Setup

1. Copy `.env.example` to `.env` and fill in your GCP project details:
```bash
cp .env.example .env
# Edit .env with your values
```

2. Build and deploy locally:
```bash
# Start Minikube
minikube start

# Build image in Minikube environment
eval $(minikube docker-env)
docker build -t reddit-app:latest .

# Deploy
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Get service URL
minikube service reddit-service --url
```

3. Deploy to GCP:
```bash
# Build and push to GCR
docker build -t gcr.io/${GCP_PROJECT_ID}/reddit-app:latest .
docker push gcr.io/${GCP_PROJECT_ID}/reddit-app:latest

# Deploy to GKE
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

## API Endpoints

- `GET /`: Welcome message
- `GET /predict`: Get sentiment prediction for Reddit comment 