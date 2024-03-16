from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel

app = FastAPI()  # Initialize your FastAPI app
model = load('boston_rf.joblib')  # Load your trained model

class HouseData(BaseModel):
    features: list  # Define the expected structure of input data for prediction

@app.post('/predict')  # Define the endpoint
def predict_price(data: HouseData):
    # Make prediction using the model
    prediction = model.predict([data.features])
    return {'predicted_price': prediction[0]}
