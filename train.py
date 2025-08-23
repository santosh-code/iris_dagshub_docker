import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import dagshub
import joblib
import os

# Initialize DagsHub logging (repo name and username)
dagshub.init("iris_dagshub_docker", "santosh.flyingmachine", mlflow=True)

# Load dataset
df = pd.read_csv("data/iris.csv")

# Drop 'Id' column if present
if "Id" in df.columns:
    df = df.drop("Id", axis=1)

# Define features and target
X = df.drop("Species", axis=1)
y = df["Species"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Track with MLflow
with mlflow.start_run():
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Log params and metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy", acc)

    # Save model locally
    os.makedirs("models", exist_ok=True)
    model_path = "models/iris_rf_model.pkl"
    joblib.dump(model, model_path)

    # Log model as artifact (works with DagsHub)
    mlflow.log_artifact(model_path)
