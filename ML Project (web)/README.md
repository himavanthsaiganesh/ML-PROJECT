# Flowstate Traffic Intelligence

Flask dashboard, dataset explorer, and congestion predictors powered by the
Hyderabad road-network data in `data/traffic_congestion_complete_dataset.csv`.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
python app.py
```

Open `http://127.0.0.1:5000`. The source CSV is loaded relative to `app.py`, not
the shell's current directory. The app caches the dataframe and checks the CSV
file signature before reusing it; replacing the CSV reloads data on the next
request. Use **Reload CSV** on the dashboard or explorer to force a reload
after replacing the source. The dashboard and explorer show rows and summaries
from that file.

## Accounts

Create an account at `/register`, then sign in at `/login`. Usernames and
password hashes are stored in `instance/users.sqlite3`; plaintext passwords
are never stored. A persistent signing key is generated in the ignored
`instance/` directory, or can be provided through the `FLASK_SECRET_KEY`
environment variable. All workspace routes require a signed-in account.

## Train and refresh

Run `python train_model.py` or use **Train / retrain models** on the Models &
evaluation page after changing the CSV. Training is not triggered by page
visits. The project CSV currently contains the supplied `india_hyderabad_traffic_dataset.csv`
records for Hyderabad, India. The dashboard, district filters, model schema, and
predictions all use this same CSV. To switch datasets, replace the project CSV
while preserving the `congestion_level` target column.

The target must be `congestion_level`; training results and the
preprocessing/model pipelines are written to `models/`. Model freshness is
checked against the current CSV content, and prediction is blocked until the
models are retrained after a dataset change.

Both estimators use a stratified train/test split and preprocessing fitted
inside their saved scikit-learn pipelines. `road_id`, `road_name`, the target,
`traffic_intensity` (a direct congestion ratio), and the duplicate
`last_maintenance_date` representation are excluded from model inputs.
Evaluation metrics and feature importance in the site are calculated from the
held-out rows produced by the real training run.

## Pages

- `/` — dataset-backed dashboard and actual records
- `/dataset` — searchable, filterable schema and rows; filtered CSV downloads
  are available from the page
- `/models` — training action, saved model evaluations, and confusion matrices
- `/predict` — schema-validated inference using a saved model pipeline
- `/health` — dataset loading status
