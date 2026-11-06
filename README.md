# Fraud Detection Pipeline on Transactional Data

End-to-end fraud detection on the **PaySim** mobile-money dataset (6,362,620 transactions, 8,213 frauds). The project covers feature engineering, a transaction-graph layer, a tuned XGBoost model, and a FastAPI service with a Streamlit dashboard that explains each prediction with SHAP.

**Stack:** Python, pandas, scikit-learn, XGBoost, SciPy, SHAP, FastAPI, Streamlit

---

## The problem

Fraud is rare (about 0.13% of PaySim transactions), so accuracy is meaningless and a model has to be judged by precision, recall and PR-AUC. A fraud system also has to decide *before* a transaction is approved, so it may only use information that exists at that moment.

## Dataset

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) (Lopez-Rojas et al., CC BY-SA 4.0) simulates mobile-money transactions. Key columns: `step` (hour), `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`.

The full file (about 493 MB) is **not committed** to this repo. Download it with `scripts/download_data.py`, or from Kaggle, and save it as `data/paysim.csv`. `data/sample_transactions.csv` holds 1,000 real held-out rows so the dashboard works without the full file.

## How it works

### 1. Removing data leakage (`fraudpipe/features.py`)
`newbalanceOrig` and `newbalanceDest` are recorded *after* a transaction. In PaySim they also behave differently on fraud rows (the sender is drained to zero and the receiver's balance often isn't updated), so they reveal the label while being unavailable at decision time. They are dropped (`LEAKY_COLS`). Running `python train.py --show-leakage` retrains with them included to show how much the score inflates.

### 2. Leakage-free features (`fraudpipe/features.py`)
Built only from pre-transaction balances and strictly past activity:

| Feature | Meaning |
|---|---|
| `balance_delta_orig` | sender's balance before minus the amount |
| `amt_to_old_orig`, `amt_to_old_dest` | amount divided by the balance before the transaction |
| `drains_account`, `overdraft` | whether the transaction empties the account or exceeds the balance |
| `orig_tx_count_24h`, `dest_tx_count_24h` | transactions by this account in the previous 24 hours (velocity) |
| `orig_time_since_last`, `dest_time_since_last` | hours since the account's previous transaction (10000 = never seen) |

### 3. Transaction-network layer (`fraudpipe/graph.py`)
Accounts are nodes and transactions are edges from sender to receiver. For every account the pipeline computes in/out degree, unique senders and receivers, degree centrality and clustering coefficient. Triangles are counted with sparse matrices, so it scales to millions of edges.

The features are **causal**: the data is cut into time blocks and each block only sees the graph built from earlier blocks, so no future information leaks in.

### 4. Model (`train.py`)
- Chronological 70 / 15 / 15 train / validation / test split.
- XGBoost tuned with `RandomizedSearchCV` using time-series cross-validation, scored by PR-AUC. Tuning runs on a 600K-row sample, then the model is refit on the full training set.
- The decision threshold is the one that maximises F1 on the validation PR curve. The test set is used once, for the final evaluation.
- Two models are trained for comparison: tabular features only, and tabular plus graph features.

### 5. Serving and explanations (`app/`, `fraudpipe/scoring.py`)
- **FastAPI** (`app/api.py`): `GET /health` and `POST /predict`, which returns fraud probability, a risk tier (HIGH, MEDIUM, LOW) and the top SHAP contributions.
- **Streamlit** (`app/dashboard.py`): score a single transaction, score the held-out sample, and view the model report. It uses the API if it is running and falls back to the local model.
- **SHAP** `TreeExplainer` explains each individual prediction.

## Results

Held-out test set: the last 15% of transactions in time, 954,393 rows with 4,008 frauds. The threshold was chosen on the validation split only.

| Model | PR-AUC | Precision | Recall | F1 | TP / FP / FN |
|---|---|---|---|---|---|
| Tabular only | 0.9995 | 0.9923 | 0.9998 | 0.9960 | 4007 / 31 / 1 |
| Tabular + graph | 0.9994 | 0.9925 | 0.9970 | 0.9948 | 3996 / 30 / 12 |

Full numbers are in `reports/metrics.json`. Plots are in `reports/pr_curve_and_ablation.png` and `reports/shap_summary.png`.

## Findings and limitations

- **The graph layer did not help on real PaySim.** The dataset has 8,169 distinct receivers across 8,213 frauds, and fraud senders never appear earlier as receivers, so there are essentially no mule accounts or fraud rings for a graph to find. On the synthetic generator (`fraudpipe/synthetic.py`), which does contain rings, the graph features improve results, which shows the layer works when ring structure exists.
- **The scores are very high because PaySim is simulated.** Fraud occurs only in TRANSFER and CASH_OUT transactions and usually empties the account, so features like `drains_account` separate the classes easily. These numbers should not be read as expected performance on real bank data.
- **Velocity features come from the caller.** At serving time the API expects per-account velocity fields in the request, which in production would come from a feature store. Missing values default to "no history".
- **Cold start.** Accounts with fewer than two historical edges get zero graph features.
- **Scoring is on demand**, per request or button click. It is not a streaming system.

## Project structure

```
fraudpipe/
  features.py     leakage handling and tabular features
  graph.py        transaction-graph features (causal)
  scoring.py      model loading, risk tiers, SHAP explanations
  synthetic.py    PaySim-style synthetic data generator (with fraud rings)
app/
  api.py          FastAPI service
  dashboard.py    Streamlit dashboard
scripts/
  download_data.py
tests/            pytest tests
models/           trained model and graph snapshot
reports/          metrics, PR curves, SHAP summary
data/             sample_transactions.csv (1,000 real held-out rows)
train.py          end-to-end training and evaluation
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py                      # full PaySim -> data/paysim.csv
python train.py --csv data/paysim.csv --n-iter 3 --graph-blocks 10 --show-leakage

uvicorn app.api:app --port 8000                      # API docs at /docs
streamlit run app/dashboard.py                       # dashboard
pytest -q
```

Trained models ship in `models/`, so the API and dashboard work without retraining.

Example request:

```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d \
 '{"step":500,"type":"TRANSFER","amount":250000,"nameOrig":"C1","oldbalanceOrg":250000,"nameDest":"C90000001","oldbalanceDest":0}'
```

For a quick smoke test without the real data: `python train.py --synthetic 200000 --n-iter 3`.

## Dataset credit

PaySim: Lopez-Rojas, Elmir and Axelsson, "PaySim: A financial mobile money simulator for fraud detection", 2016. Kaggle: `ealaxi/paysim1`.
