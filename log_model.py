import mlflow
import mlflow.tensorflow

mlflow.set_experiment("my-ml-model")

with mlflow.start_run():
    mlflow.log_param("model_type", "tensorflow")

    # Log model
    mlflow.tensorflow.log_model(model, "model")

    print("Model logged!")