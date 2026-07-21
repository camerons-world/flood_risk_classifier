# UK Flood Risk Classifier

A live risk-scoring API for Greater London river monitoring stations, framed as a short-term signal for insurance underwriting. Give it a station, and it fetches current Environment Agency readings, computes rainfall/level features on the fly, and returns a 24h flood-warning risk score with a cost-based decision flag and a SHAP explanation of what drove it.

## Why I built this

UK flood risk is the reason [Flood Re](https://www.floodre.co.uk/) exists -- the government-backed reinsurance scheme covering high flood-risk homes the standard market struggles to price. Current underwriting leans heavily on **static** risk bands (the Environment Agency's Long Term Flood Risk categories), which label an area Low/Medium/High and don't update often. That's a reasonable baseline, but it's blind to the fact that flood risk is highly dynamic -- a "Medium risk" area during a dry spell is a very different real risk after 72 hours of sustained rain.

I built this as a small-scale proof of concept for a live, granular complement to that static banding: "conditions right now suggest this area's risk is elevated above its baseline." It's not a production system -- it covers 132 monitored stations across Greater London, not every postcode, and the point is to demonstrate the *pattern* a production version would need (live ingestion -> risk scoring -> explainable decision support), not to claim insurers should deploy this tomorrow.

I've written the full narrative of how this was built, including the decisions and dead ends along the way, into [PLAN.md](PLAN.md) -- worth a read if you want the "why" behind any specific choice here.

## Live demo

**[flood-risk-classifier.onrender.com](https://flood-risk-classifier.onrender.com/)** -- login `demo` / `demo`.

That login is a demo gate, not real access control -- I've published the credentials here on purpose, since Render's free tier has no private-deployment option. It's there to keep the app off casual crawlers, not to hide it from anyone reading this README. It's on Render's free tier, so it spins down after 15 minutes idle -- the first request after a while takes about a minute to wake back up.

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
- **Station selection**: I picked 132 Greater London river-level stations, choosing ones where both APIs had matching, sufficiently deep coverage (about 88% of live stations matched on a broader sample).

## Approach

1. **Features**, computed identically at training and inference time (`src/features.py`, reused by both the training pipeline and the live API -- I did this deliberately, to avoid train/serve skew): rate-of-rise of river level (1h/6h/12h), rainfall accumulation (6h/24h/72h/7d), peak rainfall intensity (max 1h burst within a trailing 6h window -- distinguishes a short intense downpour from the same total spread out gently), current level relative to a 90-day rolling baseline, current level relative to its own trailing 24h peak, and seasonality.
2. **Target**: binary -- was a Flood Warning or Severe Flood Warning issued within the next 24h for that station's area.
3. **Split**: strictly time-based, train < 2022-07-01, validation 2022-07-01..2024-07-01 (used only to pick the operating threshold), test >= 2024-07-01 (touched exactly once, for the numbers below). I chose the cutoffs by checking positive-episode counts per window, not round fractions of rows -- flood-warning episodes are lumpy across years (2024 alone had 31 of the dataset's 197 total).
4. **Models**: a class-weighted logistic regression baseline, and a class-weighted XGBoost main model (adds `station_id` as a categorical feature -- I validated this as a real improvement for this deployment's scope, since the app only ever serves these same 132 known stations).
5. **Cost-based threshold selection** -- the centerpiece of this project, not a generic classifier demo. See below.
6. **Interpretability**: SHAP (`shap.TreeExplainer`), wired into the `/predict` response itself, not just computed offline -- and translated into plain language for a reader, not just raw feature codes.

## Cost matrix

- **Missed warning: £30,000**, charged once per missed *episode* (not per hourly row). This is the ABI's reported 2025 average UK domestic flood claim -- a real, sourced figure, up 60% year-on-year.
- **False alarm: £100**, also charged once per contiguous false-alarm episode. I couldn't find an authoritative figure for this (it's internal underwriting-ops cost, never published), so it's a stated assumption -- roughly 1-2 hours of an analyst reviewing a flagged station.

I costed both sides per *episode*, not per row, and this mattered in practice: the label is per station-hour, so a real event or a sustained false alarm shows up as a run of consecutive rows. I originally costed missed events per episode but false alarms per row, and the optimizer found a real loophole -- once `station_id` let the model learn a persistent per-station bias, it just stayed flagged on one station for ~18,000 hours straight, which the row-based costing only charged once. Capping false-alarm episodes at 24h (the same natural unit as the label's own lookback window) closed that gap.

**Threshold selection ended up recall-constrained, not pure cost minimization**: the cheapest threshold outright only caught 27% of episodes -- mathematically optimal given the cost values, but too conservative to be a useful signal in practice (false alarms are individually cheap, but they accumulate across millions of hourly decision points across 132 stations, which pushes pure cost-minimization toward an overly cautious threshold). `src/cost_threshold.py` instead picks the *cheapest threshold that still catches at least 50% of episodes on validation* -- a common real-world framing, since insurers typically carry a minimum-recall SLA alongside a cost target, not a pure cost objective.

## Results

| | Precision | Recall | F1 | F2 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|
| Baseline (logistic regression), val @ 0.5 | 0.002 | 0.745 | 0.005 | 0.012 | 0.004 | -- |
| Main (XGBoost), val @ 0.5 | 0.016 | 0.403 | 0.030 | 0.068 | 0.124 | -- |
| Main (XGBoost), **test @ recall-constrained threshold (0.46)** | 0.004 | 0.308 | 0.008 | 0.018 | 0.036 | 0.855 |

Test set confusion matrix at the chosen threshold: TN=2,328,379, FP=28,701, FN=249, TP=111. Expected cost on test: **£571,300** (vs £362,600 at the pure cost-optimal threshold, and £992,000 at a naive 0.5 threshold with row-level false-alarm costing).

**ROC-AUC of 0.855 is the number I'd lead with** -- 0.5 is a coinflip, and this is a substantial, genuine margin above it. At the episode level, this threshold catches **7 of 15** held-out flood-warning episodes (47%) -- short of the 50% target on this exact test set, which I'm reading as honest small-sample noise: I picked the threshold on validation (where it hits exactly 50% of 34 episodes) and didn't re-tune against test, since that would mean tuning on the set I'm reporting from. The pure cost-optimal threshold (0.96) catches only 4/15 (27%) at £362,600; hitting >=50% costs roughly 60% more in expectation. I've documented that tradeoff -- and three feature-engineering attempts that failed to move it (catchment-level rainfall averaging, nearest-neighbor spatial features, lag/acceleration features, all evaluated on validation, none improved on the current feature set) -- in `notebooks/02_cost_matrix_analysis.ipynb` and `PLAN.md`.

Feature set: I started with 6 features (rate-of-rise 1h/6h, rainfall 6h/24h/72h, level-vs-baseline) and iterated once, adding rate-of-rise 12h, 7-day antecedent rainfall, peak rainfall intensity, and level-vs-24h-peak -- that moved test ROC-AUC from 0.797 to 0.855. See `notebooks/03_model_development.ipynb` for the comparison.

## Limitations

- **Coverage**: 132 Greater London stations, not a UK-wide system. I checked whether more Greater London stations could be added: 94 more exist in the live network, but all lack a usable historic archive (mostly tidal barriers, flood-defence sensors, and derived/difference gauges with no equivalent in the EA Hydrology API) -- not a fixable gap with the current data sources.
- **Modest episode-level recall**: even at the recall-constrained threshold, catching half of flood episodes costs ~60% more in expectation than the pure cost-optimal policy. 24h-ahead flood prediction from local rainfall/level dynamics alone is a genuinely hard signal to extract without the numerical weather prediction and catchment hydrology models real EA/Met Office forecasts use. The model discriminates well in relative terms (ROC-AUC 0.855); recall above ~50% gets expensive fast.
- **Event count is the real bottleneck, not features**: only ~180 distinct flood-warning episodes exist across the whole training set. I tried three additional feature-engineering directions and none improved on the current model -- consistent with the model being limited by how few independent labeled events exist, not by missing signal.
- **Baseline isn't truly live**: the live flood-monitoring API only serves ~29 days of history per station, not the 90 days the baseline feature needs. `data/station_baselines.csv` is a periodically-refreshed static snapshot (`src/refresh_baselines.py`), not a live recompute -- I'm stating that plainly here rather than glossing over it.
- **Rainfall gauge mismatch**: the rainfall gauge used at training time (nearest EA Hydrology station) is often not the same physical gauge available live (most Hydrology rainfall stations aren't in the live telemetry network). Live inference falls back to the nearest *live-queryable* rainfall station instead -- a reasonable proxy, but a genuine train/serve difference worth naming.
- **Data quality**: recent months of training data carry an `Unchecked` QC flag rather than `Good` (the Environment Agency hasn't finished quality-checking them yet). I used the readings as-is; this is an accepted simplification.
- **Concept drift**: the model reflects historical patterns and won't anticipate infrastructure changes (new flood defences, culvert work) that change a catchment's real risk profile going forward.

## Notebooks

`notebooks/` has three, in reading order, each already executed with real output baked in so they're browsable without re-running anything:

- `01_eda.ipynb` -- the 132 stations, matched flood warnings, and the assembled training table (class imbalance, seasonality, feature distributions).
- `02_cost_matrix_analysis.ipynb` -- the cost values, why I costed per-episode rather than per-row, the false-alarm-costing loophole that caught, and the resulting expected-cost-vs-threshold curve.
- `03_model_development.ipynb` -- baseline vs main model, ROC/PR curves, the feature-set iteration that moved test ROC-AUC from 0.797 to 0.855, and the SHAP summary plot.

## Stack

pandas, requests, scikit-learn, XGBoost, SHAP, FastAPI, Docker, GitHub Actions, pytest, Render, Jupyter.

## Repo layout

See [PLAN.md](PLAN.md) for the full build narrative and every decision I made along the way -- data source dead ends, the cost-function bug, hyperparameter tuning notes, the Hugging Face pivot, all of it.
