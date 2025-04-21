#!/bin/bash
set -e

echo "Initializing MLflow database..."
echo "PostgreSQL URL: $POSTGRESQL_URL"

# 尝试创建数据库
mlflow db upgrade "$POSTGRESQL_URL"

echo "Database initialization completed successfully!"
