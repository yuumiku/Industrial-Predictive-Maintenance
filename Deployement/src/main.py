from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
import os

app = FastAPI(title="Maintenance Production API")

MODEL_URI = os.getenv("MODEL_URI", "models:/maintenance_model/Production")
model = mlflow.sklearn.load_model(MODEL_URI)

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    probabilities = model.predict_proba(df)
    
    return {
        "machine_id": data.get("machine_id"),
        "predicted_class": int(prediction[0]),
        "class_probabilities": {
            f"class_{i}": float(prob) 
            for i, prob in enumerate(probabilities[0])
        },
        "confidence": float(probabilities[0][prediction[0]])
    }

@app.get("/health")
def health():
    return {"status": "healthy"}