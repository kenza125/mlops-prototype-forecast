import mlflow
import pandas as pd
import torch
from chronos import BaseChronosPipeline

mlflow.set_experiment("forecasting-prototype")
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-tiny",
    device_map="cpu",
    torch_dtype=torch.float32,
)

df = pd.read_csv(
    "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
)

with mlflow.start_run():
    mlflow.log_param("model_name", "amazon/chronos-bolt-tiny")
    mlflow.log_param("prediction_length", 12)
    quantiles, mean = pipeline.predict_quantiles(
        context=torch.tensor(df["#Passengers"]),
        prediction_length=12,
        quantile_levels=[0.1, 0.5, 0.9],
    )
    median_forecast = quantiles[0, :, 1]
    mlflow.log_metric("forecast_mean_value", float(median_forecast.mean()))
    mlflow.log_metric("forecast_max_value", float(median_forecast.max()))

print("Run termine.")