from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import tensorflow as tf
import numpy as np
import logging
from typing import List
import sqlite3

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

try:
    model = tf.keras.models.load_model("model/model.h5")
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Model loading failed: {e}")
    raise

app = FastAPI(title="ML API", version="1.0")

def init_db():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input TEXT,
            output TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_prediction(input_data, output_data):
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions (input, output) VALUES (?, ?)",
        (str(input_data), str(output_data))
    )
    conn.commit()
    conn.close()

class InputData(BaseModel):
    features: List[float]

    @validator("features")
    def validate_features(cls, v):
        if len(v) != 3:  # change based on your model input size
            raise ValueError("Exactly 3 features required")
        return v

class BatchInputData(BaseModel):
    data: List[InputData]

@app.get("/")
def home():
    return {"message": "ML API is running"}

@app.post("/v1/predict")
def predict(input_data: InputData):
    try:
        features = np.array([input_data.features])
        prediction = model.predict(features)

        result = prediction.tolist()

        # Log prediction
        logging.info(f"Input: {input_data.features}, Output: {result}")

        # Save to DB
        save_prediction(input_data.features, result)

        return {"prediction": result}

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/v1/predict/batch")
def batch_predict(batch_data: BatchInputData):
    try:
        features = np.array([item.features for item in batch_data.data])
        predictions = model.predict(features)

        results = predictions.tolist()

        logging.info(f"Batch Input: {features.tolist()}, Output: {results}")

        for inp, out in zip(features.tolist(), results):
            save_prediction(inp, out)

        return {"predictions": results}

    except Exception as e:
        logging.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")