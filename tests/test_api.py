import json
from pathlib import Path

import pytest
import requests
from fastapi.testclient import TestClient

from app import ea_client
from app.main import BASIC_AUTH_PASSWORD, BASIC_AUTH_USERNAME, app

client = TestClient(app, raise_server_exceptions=False)
client.auth = (BASIC_AUTH_USERNAME, BASIC_AUTH_PASSWORD)

FIXTURES = Path(__file__).parent / "fixtures"
STATION_ID = "2893TH"
LEVEL_MEASURE_ID = "2893TH-level-stage-i-15_min-mASD"
RAINFALL_STATION_ID = "247540TP"
RAINFALL_MEASURE_ID = "247540TP-rainfall-tipping_bucket_raingauge-t-15_min-mm"
BASE_URL = "https://environment.data.gov.uk/flood-monitoring"


def _load(name):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture(autouse=True)
def _clear_ea_cache():
    ea_client._cache.clear()
    yield
    ea_client._cache.clear()


def _mock_live_ea(requests_mock):
    requests_mock.get(f"{BASE_URL}/id/stations/{STATION_ID}", json=_load("ea_station_level.json"))
    requests_mock.get(f"{BASE_URL}/id/measures/{LEVEL_MEASURE_ID}/readings", json=_load("ea_readings_level.json"))
    requests_mock.get(f"{BASE_URL}/id/stations/{RAINFALL_STATION_ID}", json=_load("ea_station_rainfall.json"))
    requests_mock.get(f"{BASE_URL}/id/measures/{RAINFALL_MEASURE_ID}/readings", json=_load("ea_readings_rainfall.json"))


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_risk_score(requests_mock):
    _mock_live_ea(requests_mock)

    response = client.get("/predict", params={"station_id": STATION_ID})

    assert response.status_code == 200
    body = response.json()
    assert body["station_id"] == STATION_ID
    assert 0.0 <= body["risk_score"] <= 1.0
    assert isinstance(body["elevated_risk"], bool)
    assert body["stale"] is False
    assert len(body["explanation"]) > 0
    assert {"feature", "value", "shap_contribution"} <= body["explanation"][0].keys()


def test_predict_unknown_station_returns_404():
    response = client.get("/predict", params={"station_id": "not-a-real-station"})
    assert response.status_code == 404


def test_predict_falls_back_to_stale_data_when_ea_is_down(requests_mock):
    _mock_live_ea(requests_mock)
    warm = client.get("/predict", params={"station_id": STATION_ID})
    assert warm.status_code == 200
    assert warm.json()["stale"] is False

    ea_client.CACHE_TTL_SECONDS, original_ttl = 0, ea_client.CACHE_TTL_SECONDS
    try:
        requests_mock.get(f"{BASE_URL}/id/measures/{LEVEL_MEASURE_ID}/readings", exc=requests.exceptions.ConnectTimeout)
        response = client.get("/predict", params={"station_id": STATION_ID})
    finally:
        ea_client.CACHE_TTL_SECONDS = original_ttl

    assert response.status_code == 200
    assert response.json()["stale"] is True


def test_predict_hard_error_when_no_cache_to_fall_back_on(requests_mock):
    requests_mock.get(f"{BASE_URL}/id/stations/{STATION_ID}", exc=requests.exceptions.ConnectTimeout)
    response = client.get("/predict", params={"station_id": STATION_ID})
    assert response.status_code == 502


def test_predict_requires_auth():
    unauthenticated = TestClient(app, raise_server_exceptions=False)
    response = unauthenticated.get("/predict", params={"station_id": STATION_ID})
    assert response.status_code == 401


def test_health_does_not_require_auth():
    unauthenticated = TestClient(app, raise_server_exceptions=False)
    response = unauthenticated.get("/health")
    assert response.status_code == 200
