import json
from pathlib import Path

import pytest
import requests
from fastapi.testclient import TestClient

from app import ea_client, postcode_client
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


# station's own coordinates, used to make a "postcode" resolve deterministically
# to STATION_ID without needing a second set of live-EA fixtures for a different station
STATION_LAT, STATION_LONG = 51.466242, -0.501779

LEVEL_HYDROLOGY_MEASURE = "6407a424-72d2-4a6d-9917-e2e5e76c5572-level-i-900-m-qualified"
RAINFALL_HYDROLOGY_MEASURE = "bdc4c2f3-8e01-4451-8608-8dd8a6f5c359-rainfall-t-900-mm-qualified"
HYDROLOGY_BASE_URL = "https://environment.data.gov.uk/hydrology"


def _mock_backtest_hydrology(requests_mock):
    requests_mock.get(f"{HYDROLOGY_BASE_URL}/id/stations.json?wiskiID={STATION_ID}", json=_load("hydrology_station_2893TH.json"))
    requests_mock.get(f"{HYDROLOGY_BASE_URL}/id/stations.json?wiskiID=310001TP", json=_load("hydrology_station_310001TP.json"))
    requests_mock.get(f"{HYDROLOGY_BASE_URL}/id/measures/{LEVEL_HYDROLOGY_MEASURE}/readings", json=_load("hydrology_readings_level.json"))
    requests_mock.get(f"{HYDROLOGY_BASE_URL}/id/measures/{RAINFALL_HYDROLOGY_MEASURE}/readings", json=_load("hydrology_readings_rainfall.json"))


def test_postcode_geocode_parses_real_response_shape(requests_mock):
    requests_mock.get("https://api.postcodes.io/postcodes/SW1A1AA", json=_load("postcode_sw1a1aa.json"))
    lat, long = postcode_client.geocode("SW1A1AA")
    assert lat == pytest.approx(51.50101)
    assert long == pytest.approx(-0.141563)


def test_predict_by_postcode_returns_nearest_station(requests_mock, monkeypatch):
    monkeypatch.setattr("app.model_wrapper.postcode_client.geocode", lambda postcode: (STATION_LAT, STATION_LONG))
    _mock_live_ea(requests_mock)

    response = client.get("/predict", params={"postcode": "SW1A 1AA"})

    assert response.status_code == 200
    body = response.json()
    assert body["station_id"] == STATION_ID
    assert body["resolved_station"]["station_id"] == STATION_ID
    assert body["resolved_station"]["distance_km"] < 0.01
    assert "nearby_stations" in body


def test_predict_postcode_and_station_id_together_returns_400():
    response = client.get("/predict", params={"station_id": STATION_ID, "postcode": "SW1A 1AA"})
    assert response.status_code == 400


def test_predict_no_params_returns_400():
    response = client.get("/predict")
    assert response.status_code == 400


def test_predict_backtest_returns_historical_response(requests_mock):
    _mock_backtest_hydrology(requests_mock)

    response = client.get("/predict", params={"station_id": STATION_ID, "date": "2024-01-15"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "backtest"
    assert body["as_of"].startswith("2024-01-15")
    assert 0.0 <= body["risk_score"] <= 1.0


def test_predict_date_out_of_range_returns_400():
    response = client.get("/predict", params={"station_id": STATION_ID, "date": "2005-01-01"})
    assert response.status_code == 400


def test_predict_bad_date_format_returns_400():
    response = client.get("/predict", params={"station_id": STATION_ID, "date": "not-a-date"})
    assert response.status_code == 400


def test_predict_postcode_service_500_returns_503_mentioning_postcode(requests_mock):
    requests_mock.get("https://api.postcodes.io/postcodes/SW1A1AA", status_code=500)
    response = client.get("/predict", params={"postcode": "SW1A1AA"})
    assert response.status_code == 503
    assert "postcode" in response.json()["detail"].lower()
    assert "EA" not in response.json()["detail"]
    assert response.headers["Retry-After"]


def test_predict_postcode_service_timeout_returns_503_mentioning_postcode(requests_mock):
    requests_mock.get("https://api.postcodes.io/postcodes/SW1A1AA", exc=requests.exceptions.ConnectTimeout)
    response = client.get("/predict", params={"postcode": "SW1A1AA"})
    assert response.status_code == 503
    assert "postcode" in response.json()["detail"].lower()
    assert "EA" not in response.json()["detail"]


def test_predict_unknown_postcode_still_returns_404(requests_mock):
    requests_mock.get("https://api.postcodes.io/postcodes/ZZ99ZZ", status_code=404)
    response = client.get("/predict", params={"postcode": "ZZ99ZZ"})
    assert response.status_code == 404
