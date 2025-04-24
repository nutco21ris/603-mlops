from fastapi import FastAPI
import uvicorn
import joblib
from pydantic import BaseModel

class Comment(BaseModel):
    reddit_comment: str

app = FastAPI(
    title="Reddit Comment Classifier",
    description="A simple API for classifying Reddit comments",
    version="1.0.0"
)

# 加载模型
model = joblib.load("data/reddit_model_pipeline.joblib")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Reddit Comment Classifier API"}

@app.post("/predict")
def predict(comment: Comment):
    prediction = model.predict_proba([comment.reddit_comment])[0]
    return {
        "remove_probability": float(prediction[1]),
        "keep_probability": float(prediction[0])
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000) 