from celery import Celery
import tensorflow as tf
import numpy as np

celery_app = Celery(
    "tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

model = tf.keras.models.load_model("model/model.h5")

@celery_app.task
def predict_task(features):
    features = np.array([features])
    prediction = model.predict(features)
    return prediction.tolist()