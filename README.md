# UK Flood Risk Classifier

A live risk-scoring API for UK river/rainfall monitoring stations, framed as a short-term signal for insurance underwriting. Static flood risk maps (like the Environment Agency's Long Term Flood Risk bands) update infrequently -- this project explores a live complement to that: "conditions right now suggest this area's risk is elevated above its baseline band."

Status: in development. See [PLAN.md](PLAN.md) for the full write-up of scope, data sources, and decisions made along the way.

## Quickstart

```bash
python -m venv flood_risk_venv
source flood_risk_venv/bin/activate
pip install -r requirements-dev.txt
pytest
uvicorn app.main:app --reload
```

## Why

UK flood risk is the reason Flood Re exists -- the government-backed reinsurance scheme covering high flood-risk homes the standard market struggles to price. This project is a small-scale version of the kind of live signal that pricing and underwriting decisions depend on. Full context in [PLAN.md](PLAN.md).
