from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel
import mlflow
import os

app = FastAPI(title="Gold Demand Prediction API")

class GoldDemandFeatures(BaseModel):
    Q1_20: float
    Q2_20: float
    Q3_20: float
    Q4_20: float

# 在启动时加载模型
model = None

@app.on_event("startup")
async def load_model():
    global model
    # 设置MLflow tracking URI为本地
    mlflow.set_tracking_uri("file:../mlruns")
    
    # 获取实验ID
    experiment_name = "gold_demand_prediction"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise Exception(f"Experiment {experiment_name} not found")
        
    # 获取最新的run
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    if len(runs) == 0:
        raise Exception("No runs found")
        
    run_id = runs.iloc[0].run_id
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

@app.post("/predict")
async def predict(features: GoldDemandFeatures):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
            
        # 转换特征为DataFrame
        feature_dict = {
            "Q1'20": features.Q1_20,
            "Q2'20": features.Q2_20,
            "Q3'20": features.Q3_20,
            "Q4'20": features.Q4_20
        }
        df = pd.DataFrame([feature_dict])
        
        # 使用模型预测
        prediction = model.predict(df)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Gold Demand Prediction API is running"} 