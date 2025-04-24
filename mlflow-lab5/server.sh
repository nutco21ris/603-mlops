#!/bin/bash
set -e

echo "Starting MLflow server..."
echo "PostgreSQL URL: $POSTGRESQL_URL"
echo "Storage URL: $STORAGE_URL"

# Start MLflow server without attempting to initialize database
exec mlflow server \
  --host 0.0.0.0 \
  --port 8080 \
  --backend-store-uri $POSTGRESQL_URL \
  --artifacts-destination $STORAGE_URL
