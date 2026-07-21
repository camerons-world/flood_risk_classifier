from fastapi import FastAPI, HTTPException

from app import model_wrapper

app = FastAPI(title="UK Flood Risk Classifier")


@app.on_event("startup")
def _load_model():
    model_wrapper.load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict")
def predict(station_id: str):
    try:
        return model_wrapper.predict(station_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="EA API unavailable") from exc
