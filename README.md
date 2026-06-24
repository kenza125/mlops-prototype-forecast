# MLOps Prototype — Time Series Forecasting

> A modular MLOps prototype for AI-based time series forecasting, built as a reference implementation for the **STORE** project at LaRINA Laboratory.  
> Uses **Amazon Chronos-Bolt-Tiny**, **FastAPI**, **MLflow**, **Docker**, and **GitHub Actions**.

---

## Overview

This repository demonstrates a full MLOps lifecycle for a forecasting model:

- **Model serving** via a REST API (FastAPI)
- **Experiment tracking** via MLflow
- **Containerization** via Docker
- **CI/CD** via GitHub Actions
- **Automated testing** via Pytest

The reference dataset used for validation is the classic **AirPassengers** time series.  
This prototype is designed to be generic and extensible to other forecasting use cases (e.g., EV charging load prediction).

---

## Project Structure

```
mlops-prototype-forecast/
├── .github/
│   └── workflows/         # GitHub Actions CI/CD pipelines
├── api.py                 # FastAPI application — /forecast endpoint
├── track.py               # MLflow experiment tracking script
├── predict.py             # Standalone prediction script
├── test_api.py            # Pytest test suite for the API
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container image definition
├── .dockerignore
└── .gitignore
```

---

## Tech Stack

| Component         | Technology                          |
|-------------------|-------------------------------------|
| Forecasting Model | `amazon/chronos-bolt-tiny` (HuggingFace) |
| API Framework     | FastAPI + Uvicorn                   |
| Experiment Tracking | MLflow                            |
| Containerization  | Docker                              |
| CI/CD             | GitHub Actions                      |
| Testing           | Pytest + HTTPX                      |

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/kenza125/mlops-prototype-forecast.git
cd mlops-prototype-forecast
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

> **Note:** A compatible version of `transformers` is required by Chronos. Use `transformers==4.45.2` if you encounter version conflicts.

### 3. Run the API

```bash
uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

---

## API Usage

### `GET /`
Health check endpoint.

**Response:**
```json
{ "message": "API en ligne. Va sur /docs pour tester." }
```

### `POST /forecast`
Generate a probabilistic forecast from a time series.

**Request body:**
```json
{
  "values": [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118],
  "prediction_length": 12
}
```

**Response:**
```json
{
  "forecast_median": [130.2, 141.5, 158.3, ...]
}
```

The endpoint returns the **median forecast** (Q50) and logs each prediction to `predictions_log.csv` with a timestamp.

Quantiles available internally: **Q10**, **Q50**, **Q90**.

---

## MLflow Experiment Tracking

Run the tracking script to log a training/inference run:

```bash
python track.py
```

This will:
- Load the AirPassengers dataset from GitHub
- Run inference with `chronos-bolt-tiny`
- Log parameters and metrics to MLflow (`forecasting-prototype` experiment)

Launch the MLflow UI:

```bash
mlflow ui
```

Open `http://localhost:5000` to explore tracked runs.

---

## Docker

### Build the image

```bash
docker build -t mlops-forecast .
```

### Run the container

```bash
docker run -p 8000:8000 mlops-forecast
```

---

## CI/CD (GitHub Actions)

The `.github/workflows/` directory contains an automated pipeline that runs on every push:

- Installs dependencies
- Runs the Pytest test suite (`test_api.py`)

This ensures the API contract is validated on each commit.

---

## Testing

```bash
pytest test_api.py -v
```

---

## Context

This prototype was developed as preparation for an internship at **LaRINA Laboratory** (ENSTAB), within the **STORE** project — a generic MLOps framework for AI-based forecasting tools targeting interoperability with partners **THI** and **Steinbacher-Consult**.

The EV charging forecaster is the first reference use case of the STORE framework.

---

## Author

**Kenza** — 2nd year engineering student, specialization in *Digitalisation et Analyse de Données* (DAD), ENSTAB  
[GitHub Profile](https://github.com/kenza125)

---

## License

This project is open source and available under the [MIT License](LICENSE).
