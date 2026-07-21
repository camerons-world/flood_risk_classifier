from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from app import model_wrapper

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_wrapper.load_model()
    yield


app = FastAPI(title="UK Flood Risk Classifier", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stations")
def stations():
    return model_wrapper.list_stations()


@app.get("/predict")
def predict(station_id: str):
    try:
        return model_wrapper.predict(station_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="EA API unavailable") from exc


@app.get("/")
def frontend():
    return FileResponse(STATIC_DIR / "index.html")
