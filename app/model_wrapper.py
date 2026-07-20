from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models" / "model.joblib"


def load_model():
    raise NotImplementedError("no trained model yet -- run src/train.py first")


def predict(station_id: str, readings: list[dict]) -> dict:
    raise NotImplementedError("model not trained yet")
