import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from service.schemas import HouseData, PredictionResponse

app = FastAPI(title='House Price Prediction Service')
model = joblib.load('model.joblib')

@app.post("/predict", response_model=PredictionResponse)
def predict(data: HouseData):  # type: ignore
    df = pd.DataFrame([data.model_dump()])
    log_prediction = model.predict(df)[0]
    real_price = np.expm1(log_prediction)
    return PredictionResponse(predicted_price=float(real_price))