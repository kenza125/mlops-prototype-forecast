from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200

def test_forecast():
    historical = [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118] * 3  # 36 points
    response = client.post("/forecast", json={"values": historical, "prediction_length": 6})
    assert response.status_code == 200
    assert "forecast_median" in response.json()
    assert len(response.json()["forecast_median"]) == 6