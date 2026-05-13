import joblib
import numpy as np
from pydantic import create_model, BaseModel
from typing import Optional

try:
    model = joblib.load('model.joblib')
    feature_names = model.feature_names_in_.tolist()
except Exception as e:
    print(f'Failed to load model: {e}')
    feature_names = []

input_fields = {name: (Optional[object], None) for name in feature_names}
HouseData = create_model('HouseData', **input_fields)

class PredictionResponse(BaseModel):
    predicted_price: float
    currency: str = "USD"
    status: str = "success"