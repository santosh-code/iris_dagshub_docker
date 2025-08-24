import os
import mlflow

# Read env vars
uri = os.getenv("MLFLOW_TRACKING_URI")
user = os.getenv("MLFLOW_TRACKING_USERNAME")
token = os.getenv("MLFLOW_TRACKING_PASSWORD")

print("Using MLflow URI:", uri)
print("Username:", user)
print("Token length:", len(token) if token else "MISSING")

# Configure MLflow
mlflow.set_tracking_uri(uri)

# Start a dummy run
with mlflow.start_run():
    mlflow.log_param("test_param", 123)
    mlflow.log_metric("test_metric", 0.99)

print("✅ Logged successfully to DagsHub MLflow!")
