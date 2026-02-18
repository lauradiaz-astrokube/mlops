from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
import mlflow.pyfunc

MODEL_URI = "models:/mlops-model@challenger"

model = mlflow.pyfunc.load_model(MODEL_URI)
app = FastAPI()

class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.post("/infer")
def infer(features: HouseFeatures):
    
    data = np.array([[
        features.MedInc,
        features.HouseAge,
        features.AveRooms,
        features.AveBedrms,
        features.Population,
        features.AveOccup,
        features.Latitude,
        features.Longitude
    ]])

    prediction = model.predict(data)

    return {
        "prediction": float(prediction[0])
    }
