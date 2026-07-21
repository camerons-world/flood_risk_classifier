# UK Flood Risk Classifier

A live risk-scoring API for Greater London river monitoring stations, framed as a short-term signal for insurance underwriting. Given a station, it fetches current Environment Agency readings, computes rainfall/level features on the fly, and returns a 24h flood-warning risk score with a cost-based decision flag and a SHAP explanation of what drove it.

## Why this exists

UK flood risk is the reason [Flood Re](https://www.floodre.co.uk/) exists -- the government-backed reinsurance scheme covering high flood-risk homes the standard market struggles to price. Current underwriting leans heavily on **static** risk bands (the Environment Agency's Long Term Flood Risk categories), which label an area Low/Medium/High and don't update often. That's a reasonable baseline, but it's blind to the fact that flood risk is highly dynamic -- a "Medium risk" area during a dry spell is a very different real risk after 72 hours of sustained rain.

This project is a small-scale proof of concept for a live, granular complement to that static banding: "conditions right now suggest this area's risk is elevated above its baseline." It's not a production system -- it covers 132 monitored stations across Greater London, not every postcode, and the point is to demonstrate the *pattern* a production version would need (live ingestion -> risk scoring -> explainable decision support), not to claim insurers should deploy this tomorrow.

Full narrative of how this was built, including the decisions and dead ends along the way, is in [PLAN.md](PLAN.md).

## Quickstart

```bash
python -m venv flood_risk_venv
source flood_risk_venv/bin/activate
pip install -r requirements-dev.txt
pytest
uvicorn app.main:app --reload
```

Then open `http://localhost:8000` for the frontend, or query the API directly: `GET /predict?station_id=2893TH`.

## Data

- **Training**: [EA Hydrology API](https://environment.data.gov.uk/hydrology/) -- 15-minute river level/flow and rainfall readings back to 2010 (and often much further), used to build a 2022-07 cutoff training set. This is a different API from the live one below; its "qualified" (QC'd) data lags the present by roughly a year, which is fine for training but not for live inference.
- **Live inference**: [EA flood-monitoring API](https://environment.data.gov.uk/flood-monitoring/) -- the real-time telemetry network, updated roughly every 15 minutes, no API key required. Only serves ~29 days of history per station, which is enough for the rainfall/rate-of-rise features but not the 90-day baseline (see Limitations).
- **Labels**: the EA's Historic Flood Warnings dataset (2006-present), matched to the nearest station by spatial join since warning areas don't carry a direct station ID.
- Station selection: 132 Greater London river-level stations, chosen where both APIs had matching, sufficiently deep coverage (about 88% of live stations matched on a broader sample).

## Approach

1. **Features**, computed identically at training and inference time (`src/features.py`, reused by both the training pipeline and the live API -- this is deliberate, to avoid train/serve skew): rate-of-rise of river level (1h, 6h), rainfall accumulation (6h/24h/72h), current level relative to a 90-day rolling baseline, and seasonality.
2. **Target**: binary -- was a Flood Warning or Severe Flood Warning issued within the next 24h for that station's area.
3. **Split**: strictly time-based, train < 2022-07-01, validation 2022-07-01..2024-07-01 (used only to pick the operating threshold), test >= 2024-07-01 (touched exactly once, for the numbers below). Cutoffs were chosen by checking positive-episode counts per window, not round fractions of rows -- flood-warning episodes are lumpy across years (2024 alone had 31 of the dataset's 197 total).
4. **Models**: a class-weighted logistic regression baseline, and a class-weighted XGBoost main model (adds `station_id` as a categorical feature -- a real, validated improvement for this deployment's scope, since the app only ever serves these same 132 known stations).
5. **Cost-based threshold selection** -- the centerpiece of this project, not a generic classifier demo. See below.
6. **Interpretability**: SHAP (`shap.TreeExplainer`), wired into the `/predict` response itself, not just computed offline.

## Cost matrix

- **Missed warning: £30,000**, charged once per missed *episode* (not per hourly row). This is the ABI's reported 2025 average UK domestic flood claim -- a real, sourced figure, up 60% year-on-year.
- **False alarm: £100**, also charged once per contiguous false-alarm episode. No authoritative figure exists for this (it's internal underwriting-ops cost, never published) -- this is a stated assumption, roughly 1-2 hours of an analyst reviewing a flagged station.

Both sides are costed per *episode*, not per row, and this mattered in practice: the label is per station-hour, so a real event or a sustained false alarm shows up as a run of consecutive rows. An earlier version of this costed missed events per episode but false alarms per row, and the optimizer found a real loophole -- once `station_id` let the model learn a persistent per-station bias, it just stayed flagged on one station for ~18,000 hours straight, which the row-based costing only charged once. Capping false-alarm episodes at 24h (the same natural unit as the label's own lookback window) closed that gap.

## Results

| | Precision | Recall | F1 | F2 | PR-AUC |
|---|---|---|---|---|---|
| Baseline (logistic regression), val @ 0.5 | 0.002 | 0.727 | 0.005 | 0.012 | 0.004 |
| Main (XGBoost), val @ 0.5 | 0.010 | 0.432 | 0.020 | 0.047 | 0.096 |
| Main (XGBoost), **test @ cost-optimal threshold (0.94)** | 0.018 | 0.208 | 0.034 | 0.068 | 0.044 |

Test set confusion matrix at the chosen threshold: TN=2,353,724, FP=4,012, FN=285, TP=75. Expected cost on test: **£375,000**, down from £992,000 at a naive 0.5 threshold with row-level false-alarm costing.

PR-AUC of 0.044 looks small in isolation, but the positive base rate on test is 0.015% -- so this is about **286x** the base rate, real (if modest) ranking power. At the episode level, the model catches **4 of 15** held-out flood-warning episodes (27%) at its cost-optimal threshold. That's the honest headline number: the cost-optimal policy, given how much false alarms accumulate across millions of hourly decision points, ends up conservative.

## Limitations

- **Coverage**: 132 Greater London stations, not a UK-wide system. A real deployment would need this pattern repeated (and validated) across every EA region.
- **Modest recall**: 27% episode catch rate at the cost-optimal threshold is a real limitation, not a rounding error -- 24h-ahead flood prediction from 9 basic features (mostly local rainfall/level dynamics) is a hard signal to extract without the numerical weather prediction and catchment hydrology models real EA/Met Office forecasts use.
- **Baseline isn't truly live**: the live flood-monitoring API only serves ~29 days of history per station, not the 90 days the baseline feature needs. `data/station_baselines.csv` is a periodically-refreshed static snapshot (`src/refresh_baselines.py`), not a live recompute -- stated here rather than glossed over.
- **Rainfall gauge mismatch**: the rainfall gauge used at training time (nearest EA Hydrology station) is often not the same physical gauge available live (most Hydrology rainfall stations aren't in the live telemetry network). Live inference falls back to the nearest *live-queryable* rainfall station instead -- a reasonable proxy, but a genuine train/serve difference worth naming.
- **Data quality**: recent months of training data carry an `Unchecked` QC flag rather than `Good` (the Environment Agency hasn't finished quality-checking them yet). Readings are still real and used as-is; this is an accepted simplification.
- **Concept drift**: the model reflects historical patterns and won't anticipate infrastructure changes (new flood defences, culvert work) that change a catchment's real risk profile going forward.

## Stack

pandas, requests, scikit-learn, XGBoost, SHAP, FastAPI, Docker, GitHub Actions, pytest, Hugging Face Spaces.

## Repo layout

See [PLAN.md](PLAN.md) for the full build narrative and every decision made along the way -- data source dead ends, the cost-function bug, hyperparameter tuning notes, all of it.
