# UK Flood Risk Classifier

A live risk-scoring API for Greater London river monitoring stations, framed as a short-term signal for insurance underwriting. Give it a station -- or a postcode, resolved to the nearest one -- and it fetches current Environment Agency readings, computes rainfall/level features on the fly, and returns a 24h flood-warning risk score with a cost-based decision flag and a SHAP explanation of what drove it. You can also check a past date instead of "now," using real historical readings from that day.

## Why I built this

UK flood risk is the reason [Flood Re](https://www.floodre.co.uk/) exists -- the government-backed reinsurance scheme for high flood-risk homes the standard market can't price well. Underwriting still leans on static risk bands that update infrequently and miss short-term dynamics: a "Medium risk" postcode in a dry spell is a different real risk after 72 hours of rain. I built this as a proof of concept for the missing live signal -- "conditions right now suggest this area's risk is elevated above its baseline."

It's not a production system -- 132 monitored stations across Greater London, not every postcode. The point is to demonstrate the *pattern* a production version would need, not to claim insurers should deploy this tomorrow.

Full narrative of how this was built, decisions and dead ends included, is in [PLAN.md](PLAN.md).

## Live demo

**[flood-risk-classifier.onrender.com](https://flood-risk-classifier.onrender.com/)** -- login `demo` / `demo`.

That login is a demo gate, not real access control -- I've published the credentials here on purpose. It's on Render's free tier, so it spins down after 15 minutes idle -- the first request after a while takes about a minute to wake back up.

## Quickstart

```bash
python -m venv flood_risk_venv
source flood_risk_venv/bin/activate
pip install -r requirements-dev.txt
pytest
uvicorn app.main:app --reload
```

Then open `http://localhost:8000` for the frontend, or query the API directly:
- `GET /predict?station_id=2893TH` -- live risk for a known station
- `GET /predict?postcode=SW1A+1AA` -- live risk for the nearest station to a UK postcode
- `GET /predict?station_id=2893TH&date=2024-01-15` -- historical backtest instead of live (works with `postcode` too)

## Data

I train on the [EA Hydrology API](https://environment.data.gov.uk/hydrology/) -- deep-history river level/flow and rainfall, back to 2010, cut off at 2022-07 for training. Live inference hits a different, shallower API instead ([flood-monitoring](https://environment.data.gov.uk/flood-monitoring/), ~29 days of history, updated every 15 minutes) -- fine for the rate-of-rise and rainfall features, not enough for the 90-day baseline (more on that in Limitations). Labels come from the EA's Historic Flood Warnings dataset, spatially joined to the nearest station since warnings don't carry a station ID.

I picked 132 Greater London stations myself, keeping only ones where both APIs had solid matching coverage. Postcodes get geocoded through [postcodes.io](https://postcodes.io) (free, keyless, same deal as everything else here). Backtests just reuse the Hydrology API for a past window instead of "now" -- not a new data source, just a different ask of the same one.

## Approach

1. **Features** live in `src/features.py` and get reused by training and the live API alike, on purpose -- I didn't want train/serve skew creeping in. Rate-of-rise (1h/6h/12h), rainfall accumulation (6h/24h/72h/7d), peak rainfall intensity, level vs. a 90-day baseline, level vs. its own 24h peak, and seasonality.
2. **Target**: binary -- a Flood Warning or Severe Flood Warning within the next 24h for that station.
3. **Split**: time-based only. Train < 2022-07, validate 2022-07 to 2024-07 (picks the threshold), test from 2024-07 (touched once). I set the cutoffs by checking how many flood episodes fell in each window, not by round fractions -- episodes are lumpy year to year.
4. **Models**: class-weighted logistic regression as a baseline, class-weighted XGBoost as the real one. XGBoost gets `station_id` as a feature too -- a genuine improvement here, since the app only ever serves these same 132 stations.
5. **Cost-based threshold selection** -- the actual point of this project. £30,000 per missed warning (sourced from the ABI's average UK flood claim), £100 per false alarm (my own assumption, nothing public exists for this), both charged per episode rather than per row. Picked to hit 50% recall on validation rather than just minimizing cost outright -- full reasoning in `PLAN.md`.
6. **Interpretability**: SHAP, wired into the response itself and translated into plain language, not raw feature codes.
7. **Postcode lookup**: resolves to the nearest station, plus 2 more nearby shown individually rather than averaged. Reasoning (why not H3, why not average) is in `PLAN.md`.
8. **Historical backtest**: an optional `date` param replaces live data with real historical readings -- deliberately not a forecast. Why, in `PLAN.md`.

## Results

| | Precision | Recall | F1 | F2 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|
| Baseline (logistic regression), val @ 0.5 | 0.002 | 0.745 | 0.005 | 0.012 | 0.004 | -- |
| Main (XGBoost), val @ 0.5 | 0.016 | 0.403 | 0.030 | 0.068 | 0.124 | -- |
| Main (XGBoost), **test @ recall-constrained threshold (0.46)** | 0.004 | 0.308 | 0.008 | 0.018 | 0.036 | 0.855 |

## Limitations

- **Coverage**: 132 Greater London stations, not the UK. I checked -- 94 more stations exist in the area but none have a usable archive (mostly tidal barriers and defence sensors), so this isn't a gap I can close with what's available.
- **Recall tops out**: even the recall-constrained threshold only gets to 47% on test, at real cost. 24h-ahead flood prediction from local rainfall/level alone is genuinely hard without the weather-forecasting models the Met Office actually uses. Ranking (ROC-AUC 0.855) is solid; turning that into high recall cheaply isn't.
- **~180 flood episodes, total**: that's the real ceiling, not the features. Three extra feature ideas went nowhere, which is consistent with an events problem, not a signal problem.
- **Baseline isn't truly live**: the live API only keeps ~29 days, not the 90 the baseline needs, so it's a periodically-refreshed snapshot (`src/refresh_baselines.py`) rather than a live number. Backtests don't have this problem -- they pull real history each time.
- **Live rainfall gauge differs from training's**: most training-time gauges aren't in the live telemetry network, so live inference substitutes the nearest one that is. A real, if minor, train/serve gap -- backtests avoid it entirely by using the same gauge training did.
- **Recent data is `Unchecked`, not `Good`**: the EA hasn't finished QC on the last few months. I use it as-is.
- **Concept drift**: new flood defences or drainage work will make this stale in ways it can't detect.
- **Postcode/date bounds**: postcodes past 15km from any station get a clear "no coverage," not a misleading distant match. Backtests run 2010-04-01 to yesterday, bounded by baseline history and data completeness.

## Repo layout

Full file tree and every decision behind it is in [PLAN.md](PLAN.md) -- dead ends included.
