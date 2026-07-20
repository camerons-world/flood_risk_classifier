from fastapi import FastAPI, HTTPException

from app import ea_client, model_wrapper

app = FastAPI(title="UK Flood Risk Classifier")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict")
def predict(station_id: str):
    try:
        readings = ea_client.get_latest_readings(station_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="EA API unavailable") from exc

    return model_wrapper.predict(station_id, readings)
