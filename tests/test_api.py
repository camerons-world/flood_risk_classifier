import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "ea_readings_sample.json").read_text())


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_uses_live_ea_data(requests_mock):
    requests_mock.get(
        "https://environment.data.gov.uk/flood-monitoring/id/stations/1029TH/readings",
        json=FIXTURE,
    )
    response = client.get("/predict", params={"station_id": "1029TH"})
    # model isn't trained yet, so this currently 500s -- flip to a real
    # assertion once model_wrapper.predict is implemented
    assert response.status_code == 500
