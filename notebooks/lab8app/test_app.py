import requests
import pandas as pd

df = pd.read_csv('gold_demand.csv')

# Get first row as test sample
test_data = {
    "Q1_20": float(df["Q1'20"].iloc[0]),
    "Q2_20": float(df["Q2'20"].iloc[0]),
    "Q3_20": float(df["Q3'20"].iloc[0]),
    "Q4_20": float(df["Q4'20"].iloc[0])
}

def test_prediction():
    response = requests.get("http://localhost:8080/")
    print("Root endpoint response:", response.json())
    
    response = requests.post("http://localhost:8080/predict", json=test_data)
    print("\nPrediction endpoint response:", response.json())
    print("\nTest data used:", test_data)

if __name__ == "__main__":
    test_prediction() 