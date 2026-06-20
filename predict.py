import pandas as pd
import torch
from chronos import BaseChronosPipeline

def get_pipeline():
    return BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-tiny",
        device_map="cpu",
        torch_dtype=torch.float32,
    )

if __name__ == "__main__":
    pipeline = get_pipeline()

    # Donnees historiques reelles : nombre de passagers aeriens par mois (1949-1960)
    df = pd.read_csv(
        "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
    )

    quantiles, mean = pipeline.predict_quantiles(
        context=torch.tensor(df["#Passengers"]),
        prediction_length=12,
        quantile_levels=[0.1, 0.5, 0.9],
    )

    print("Prevision (mediane) pour les 12 prochains mois :")
    print(quantiles[0, :, 1])
    