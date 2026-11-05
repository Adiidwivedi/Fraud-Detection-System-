# Fraud Detection Pipeline on Transactional Data
*Python, XGBoost, FastAPI, Streamlit, SHAP*

End-to-end fraud detection on the **PaySim** mobile-money dataset (**6,362,620 transactions**, 8,213 frauds).

| Resume claim | Where it lives |
|---|---|
| End-to-end pipeline on 6.3M+ PaySim records | `scripts/download_data.py` → `train.py` (real data, not synthetic) |
| Identified & removed post-transaction balance fields causing **data leakage**; replaced with balance-delta ratios, per-account transaction velocity, time-since-last-transaction | `fraudpipe/features.py` (`LEAKY_COLS`, `balance_delta_orig`, `amt_to_old_*`, `*_tx_count_24h`, `*_time_since_last`) |
| **Transaction-network layer**: sender-receiver graph, degree centrality + clustering coefficient | `fraudpipe/graph.py` (sparse-matrix triangles; built from history only, no look-ahead) |
| **XGBoost** tuned with `RandomizedSearchCV`; threshold chosen by sweeping precision-recall / F1 on a held-out validation set | `train.py` (`fit_xgb`, `best_f1_threshold`), temporal 70/15/15 split |
| **FastAPI** service + **Streamlit** dashboard with risk tier, fraud probability and per-transaction **SHAP TreeExplainer** explanations | `app/api.py`, `app/dashboard.py`, `fraudpipe/scoring.py` |

## Results (real PaySim, held-out last 15% of transactions = 954,393 rows, 4,008 frauds)
Threshold picked on the validation split only. 3 random-search iterations (see `--n-iter`; raise for a better search).

| Model | PR-AUC | Precision | Recall | F1 | TP / FP / FN |
|---|---|---|---|---|---|
| Tabular only | 0.9995 | 0.9923 | 0.9998 | 0.9960 | 4007 / 31 / 1 |
| Tabular + graph | 0.9994 | 0.9925 | 0.9970 | 0.9948 | 3996 / 30 / 12 |

Full numbers: `reports/metrics.json`; plots: `reports/pr_curve_and_ablation.png`, `reports/shap_summary.png`.


1. **Fraud rate:** the real PaySim file has **8,213 / 6,362,620 = 0.129%** fraud overall (not 0.82%). The fraud rate rises over the simulation clock (~0.11% in steps 1-100 to ~0.88% after step 500), so the chronological test split is ~0.42% fraud.
2. **Graph layer:** on real PaySim the graph features did **not** improve recall over the tabular baseline (see table). The reason is in the data: 8,169 distinct receivers across 8,213 frauds, and fraud cash-out senders never appear earlier as receivers, i.e. PaySim contains essentially no mule reuse / rings for a graph to find. Don't claim an improvement from these results; it's an honest ablation. (On the synthetic generator in `fraudpipe/synthetic.py`, which does contain rings, the graph layer lifts PR-AUC, which shows the layer works when ring structure exists.)
3. Leakage: PaySim's `newbalanceOrig` / `newbalanceDest` are written after the transaction; `python train.py --show-leakage` scores a model with them to show the inflation.
4. Graph features are strictly causal: rows in time block *b* only see edges from blocks < *b* (a batch-refreshed history snapshot). `models/node_features.parquet` keeps accounts with ≥2 historical edges; others are treated as cold-start (zeros).
5. The API takes per-account velocity fields from the caller (a feature store in production).

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py                        # 493 MB real PaySim -> data/paysim.csv
python train.py --csv data/paysim.csv --n-iter 8 --show-leakage   # ~10-30 min
uvicorn app.api:app --port 8000                        # API docs at /docs
streamlit run app/dashboard.py                         # dashboard
pytest -q
```
`models/` and `data/sample_transactions.csv` (1,000 real held-out PaySim rows) ship pre-built, so the API and dashboard work immediately.
```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d \
 '{"step":500,"type":"TRANSFER","amount":250000,"nameOrig":"C1","oldbalanceOrg":250000,"nameDest":"C90000001","oldbalanceDest":0}'
```
Dataset: PaySim (Lopez-Rojas et al.), Kaggle `ealaxi/paysim1`, CC BY-SA 4.0.
