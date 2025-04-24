#!/bin/bash
set -e

echo "Initializing MLflow database..."
echo "PostgreSQL URL: $POSTGRESQL_URL"

# Attempt to create database
mlflow db upgrade "$POSTGRESQL_URL"

echo "Database initialization completed successfully!"
