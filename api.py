from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
from chronos import BaseChronosPipeline
import csv
from datetime import datetime

app = FastAPI(title="API de prevision (forecasting)")
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-tiny",
    device_map="cpu",
    torch_dtype=torch.float32,
)

class SeriesInput(BaseModel):
    values: List[float]
    prediction_length: int = 12

@app.get("/")
def root():
    return {"message": "API en ligne. Va sur /docs pour tester."}

@app.post("/forecast")
def forecast(input: SeriesInput):
    quantiles, mean = pipeline.predict_quantiles(
        context=torch.tensor(input.values),
        prediction_length=input.prediction_length,
        quantile_levels=[0.1, 0.5, 0.9],
    )
    median_forecast = quantiles[0, :, 1].tolist()
    with open("predictions_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), len(input.values), input.prediction_length, median_forecast])
    return {"forecast_median": median_forecast}