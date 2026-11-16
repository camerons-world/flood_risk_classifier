import datetime as dt
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app import model_wrapper
from app.postcode_client import PostcodeLookupError
from app.model_wrapper import BACKTEST_MIN_DATE

# Seconds clients are told to wait before retrying after a postcode-service
# failure (sent as the Retry-After header). Override with the env var.
POSTCODE_RETRY_AFTER_SECONDS = int(os.environ.get("POSTCODE_RETRY_AFTER_SECONDS", "30"))
STATIC_DIR = Path(__file__).parent / "static"

# Demo login only, not real access control -- credentials are published in
# README.md on purpose. Render's free tier has no private-space toggle, so
# this exists to put a small gate in front of casual link-following, not to
# actually restrict who can see the app.
BASIC_AUTH_USERNAME = os.environ.get("BASIC_AUTH_USERNAME", "demo")
BASIC_AUTH_PASSWORD = os.environ.get("BASIC_AUTH_PASSWORD", "demo")

security = HTTPBasic()


def require_auth(credentials: Annotated[HTTPBasicCredentials, Depends(security)]) -> None:
    valid_username = secrets.compare_digest(credentials.username, BASIC_AUTH_USERNAME)
    valid_password = secrets.compare_digest(credentials.password, BASIC_AUTH_PASSWORD)
    if not (valid_username and valid_password):
        raise HTTPException(status_code=401, detail="Invalid credentials", headers={"WWW-Authenticate": "Basic"})


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_wrapper.load_model()
    yield


app = FastAPI(title="UK Flood Risk Classifier", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stations", dependencies=[Depends(require_auth)])
def stations():
    return model_wrapper.list_stations()


@app.get("/predict", dependencies=[Depends(require_auth)])
def predict(station_id: str | None = None, postcode: str | None = None, date: str | None = None):
    if bool(station_id) == bool(postcode):
        raise HTTPException(status_code=400, detail="provide exactly one of station_id or postcode")

    parsed_date = None
    if date is not None:
        try:
            parsed_date = dt.date.fromisoformat(date)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="date must be in YYYY-MM-DD format") from exc
        if not (BACKTEST_MIN_DATE <= parsed_date < dt.date.today()):
            raise HTTPException(
                status_code=400, detail=f"date must be between {BACKTEST_MIN_DATE.isoformat()} and yesterday"
            )

    try:
        return model_wrapper.predict(station_id=station_id, postcode=postcode, date=parsed_date)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PostcodeLookupError as exc:
        raise HTTPException(
            status_code=503,
            detail="Postcode lookup service unavailable, please try again shortly or pick a station instead",
            headers={"Retry-After": str(POSTCODE_RETRY_AFTER_SECONDS)},
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="EA API unavailable") from exc


@app.get("/", dependencies=[Depends(require_auth)])
def frontend():
    return FileResponse(STATIC_DIR / "index.html")
