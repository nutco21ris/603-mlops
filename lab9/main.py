from fastapi import FastAPI
import random

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "Welcome to Reddit Classifier API"}

@app.get("/predict")
def predict_sentiment():
    # 模拟预测结果
    sentiments = ["positive", "negative", "neutral"]
    prediction = random.choice(sentiments)
    confidence = round(random.random(), 2)
    
    return {
        "prediction": prediction,
        "confidence": confidence
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 