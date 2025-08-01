from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load model and scaler
model = joblib.load(r"c:\Users\user\OneDrive\Desktop\anamoly_detection\naivebayes_model.pkl")
scaler = joblib.load(r"c:\Users\user\OneDrive\Desktop\anamoly_detection\scaler.pkl")  # Optional, use only if you used a scaler

# Define feature input structure
class EquipmentData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float
    feature5: int
    feature6: int
    # Add more fields depending on your dataset

# Initialize FastAPI app
app = FastAPI(title="Industrial Equipment Anomaly Detection API")

@app.post("/predict")
def predict(data: EquipmentData):
    # Convert input to NumPy array
    input_data = np.array([[data.feature1, data.feature2, data.feature3, data.feature4, data.feature5, data.feature6]])
    
    # Scale the input (if scaler was used during training)
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    prediction_label = "Anomaly" if prediction == 1 else "Normal"

    return {
        "prediction": int(prediction),
        "label": prediction_label
    }
