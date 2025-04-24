import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import mlflow

# 设置MLflow
mlflow.set_tracking_uri("file:../mlruns")

# 创建实验
experiment_name = "gold_demand_prediction"
try:
    experiment_id = mlflow.create_experiment(experiment_name)
except:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

mlflow.set_experiment(experiment_name)

# 加载数据
df = pd.read_csv('gold_demand.csv')

# 准备特征和目标
X = df[["Q1'20", "Q2'20", "Q3'20", "Q4'20"]]
y = df["Q1'21"]  # 预测下一个季度

# 训练测试分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
with mlflow.start_run(experiment_id=experiment_id) as run:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 评估模型
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    # 记录参数和指标
    mlflow.log_params({
        "n_estimators": 100,
        "random_state": 42
    })
    mlflow.log_metrics({
        "train_r2": train_score,
        "test_r2": test_score
    })
    
    # 保存模型
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="gold_demand_final_model"
    )
    
    print(f"Training R2: {train_score:.4f}")
    print(f"Test R2: {test_score:.4f}")
    print(f"Model registered as: gold_demand_final_model")
    print(f"Run ID: {run.info.run_id}")
    print(f"Experiment ID: {experiment_id}")