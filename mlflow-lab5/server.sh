#!/bin/bash
set -e

echo "Starting MLflow server..."
echo "PostgreSQL URL: $POSTGRESQL_URL"
echo "Storage URL: $STORAGE_URL"

# 启动MLflow服务器，不再尝试初始化数据库
exec mlflow server \
  --host 0.0.0.0 \
  --port 8080 \
  --backend-store-uri $POSTGRESQL_URL \
  --artifacts-destination $STORAGE_URL
