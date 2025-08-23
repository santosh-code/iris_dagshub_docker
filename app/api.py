from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("models/model.pkl")

@app.get("/")
def root():
    return {"message": "Iris Model API is running!"}

@app.post("/predict")
def predict(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                      columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    prediction = model.predict(df)[0]
    return {"prediction": prediction}
