"""Flask application for exploring and predicting traffic congestion."""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import sqlite3
import threading
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterator

import joblib
import pandas as pd
from flask import Flask, abort, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from train_model import DATASET_PATH, MODEL_DIR, TARGET, TrainingError, train_and_save


PROJECT_DIR = Path(__file__).resolve().parent
app = Flask(__name__)


def _load_or_create_secret() -> str:
    secret_path = Path(app.instance_path) / "session_secret_key"
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        secret = secret_path.read_text(encoding="ascii").strip()
        if len(secret) >= 64:
            return secret
        raise RuntimeError(f"Session secret file is invalid: {secret_path}")
    except FileNotFoundError:
        secret = secrets.token_hex(32)
        try:
            with secret_path.open("x", encoding="ascii") as destination:
                destination.write(secret)
        except FileExistsError:
            secret = secret_path.read_text(encoding="ascii").strip()
            if len(secret) < 64:
                raise RuntimeError(f"Session secret file is invalid: {secret_path}")
        return secret


app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY") or _load_or_create_secret()
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
)
AUTH_DB_PATH = Path(app.instance_path) / "users.sqlite3"
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")


@contextmanager
def _connect_auth_db() -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(AUTH_DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _initialize_auth_db() -> None:
    AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect_auth_db() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


_initialize_auth_db()


@app.before_request
def require_login_and_protect_forms():
    endpoint = request.endpoint
    if endpoint in {"static"} or endpoint is None:
        return None
    if endpoint in {"login", "register"} and "user_id" in session:
        return redirect(url_for("dashboard"))
    if endpoint not in {"login", "register"} and "user_id" not in session:
        return redirect(url_for("login", next=request.full_path.rstrip("?")))
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        submitted_token = request.form.get("csrf_token")
        if request.is_json:
            submitted_token = (request.get_json(silent=True) or {}).get("csrf_token")
        expected_token = session.get("csrf_token", "")
        if not expected_token or not secrets.compare_digest(
            str(submitted_token or ""), expected_token
        ):
            abort(400, description="The form expired or could not be verified. Refresh and try again.")
    return None


class DatasetError(RuntimeError):
    """Raised when the project dataset is unavailable or invalid."""


class DatasetStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._frame: pd.DataFrame | None = None
        self._signature: tuple[int, int] | None = None
        self._lock = threading.RLock()

    def load(self, force: bool = False) -> pd.DataFrame:
        with self._lock:
            if not self.path.is_file():
                self._frame = None
                self._signature = None
                raise DatasetError(
                    f"Dataset not found at {self.path}. Put the original CSV at "
                    "data/traffic_congestion_complete_dataset.csv."
                )
            try:
                info = self.path.stat()
                signature = (info.st_mtime_ns, info.st_size)
                if not force and self._frame is not None and signature == self._signature:
                    return self._frame
                frame = pd.read_csv(self.path)
                if frame.empty:
                    raise DatasetError("The dataset CSV contains no records.")
                if len(frame.columns) == 0:
                    raise DatasetError("The dataset CSV contains no columns.")
                if TARGET not in frame.columns:
                    raise DatasetError(
                        f"The dataset is missing the required '{TARGET}' column."
                    )
                self._frame = frame
                self._signature = signature
                return frame
            except DatasetError:
                self._frame = None
                self._signature = None
                raise
            except (OSError, pd.errors.ParserError, UnicodeError, ValueError) as exc:
                self._frame = None
                self._signature = None
                raise DatasetError(f"Could not load the dataset CSV: {exc}") from exc


dataset_store = DatasetStore(DATASET_PATH)
_model_cache: dict[str, Any] = {}
_model_signature: tuple[int, int] | None = None
_model_lock = threading.RLock()
_recent_predictions: list[dict[str, Any]] = []
_prediction_lock = threading.Lock()


def _display_label(value: Any) -> str:
    return {"Medium": "Moderate"}.get(str(value), str(value))


def _filtered_data(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame
    search = request.args.get("search", "").strip()
    if search:
        text = result.astype("string").fillna("")
        mask = text.apply(lambda column: column.str.contains(search, case=False, regex=False)).any(axis=1)
        result = result.loc[mask]
    for column, parameter in (
        ("road_type", "road_type"),
        (TARGET, "congestion_level"),
        ("district", "district"),
    ):
        value = request.args.get(parameter, "").strip()
        if value and column in result.columns:
            result = result.loc[result[column].astype(str) == value]
    return result


def _model_status(frame: pd.DataFrame) -> tuple[bool, str]:
    results_path = MODEL_DIR / "training_results.json"
    if not results_path.is_file():
        return False, "Models have not been trained yet."
    try:
        metadata = json.loads(results_path.read_text(encoding="utf-8"))
        import hashlib

        digest = hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()
        if digest != metadata.get("dataset_sha256"):
            return False, "The CSV has changed since training. Retrain to refresh the models."
        if metadata.get("rows") != len(frame):
            return False, "The dataset row count changed. Retrain to refresh the models."
        return True, f"Trained on {metadata['rows']:,} dataset records."
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return False, f"Could not verify trained model metadata: {exc}"


def _load_models() -> tuple[dict[str, Any], dict[str, Any]]:
    global _model_signature
    paths = {
        "random_forest": MODEL_DIR / "random_forest.joblib",
        "gradient_boosting": MODEL_DIR / "gradient_boosting.joblib",
    }
    results_path = MODEL_DIR / "training_results.json"
    missing = [str(path) for path in [*paths.values(), results_path] if not path.is_file()]
    if missing:
        raise DatasetError(
            "Trained models are unavailable. Use the model page to train them from the dataset."
        )
    signature = tuple(
        path.stat().st_mtime_ns for path in [*paths.values(), results_path]
    )
    with _model_lock:
        if _model_signature == signature and _model_cache:
            metadata = json.loads(results_path.read_text(encoding="utf-8"))
            return _model_cache, metadata
        try:
            loaded = {name: joblib.load(path) for name, path in paths.items()}
            metadata = json.loads(results_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, EOFError, json.JSONDecodeError) as exc:
            raise DatasetError(f"Could not load trained models: {exc}") from exc
        _model_cache.clear()
        _model_cache.update(loaded)
        _model_signature = signature
        return _model_cache, metadata


def _prediction_input(payload: dict[str, Any], metadata: dict[str, Any]) -> pd.DataFrame:
    schema = metadata.get("input_schema", {})
    missing = [column for column in schema if column not in payload or payload[column] == ""]
    if missing:
        raise ValueError(f"Missing required input values: {', '.join(missing)}.")
    values: dict[str, Any] = {}
    for column, definition in schema.items():
        value = payload[column]
        if definition["kind"] == "numeric":
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"'{column}' must be a valid number.") from exc
            if not math.isfinite(numeric_value):
                raise ValueError(f"'{column}' must be a finite number.")
            values[column] = numeric_value
        else:
            value = str(value)
            options = definition.get("options", [])
            if options and value not in options:
                raise ValueError(f"'{column}' must be one of the available dataset values.")
            values[column] = value
    return pd.DataFrame([values], columns=list(schema))


@app.context_processor
def inject_navigation() -> dict[str, str]:
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return {
        "active_endpoint": request.endpoint or "",
        "csrf_token": token,
        "current_username": session.get("username"),
    }


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    username = ""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        with _connect_auth_db() as connection:
            user = connection.execute(
                "SELECT id, username, password_hash FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        if (
            user is None
            or len(password) > 128
            or not check_password_hash(user["password_hash"], password)
        ):
            error = "Username or password is incorrect."
        else:
            session.clear()
            session.permanent = True
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["csrf_token"] = secrets.token_urlsafe(32)
            destination = request.args.get("next", "")
            if destination.startswith("/") and not destination.startswith("//"):
                return redirect(destination)
            return redirect(url_for("dashboard"))
    return render_template("auth.html", mode="login", error=error, username=username)


@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    username = ""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirmation = request.form.get("confirm_password", "")
        if not USERNAME_PATTERN.fullmatch(username):
            error = "Username must be 3–32 characters and use letters, numbers, dots, hyphens, or underscores."
        elif len(password) < 8 or len(password) > 128:
            error = "Password must be between 8 and 128 characters."
        elif password != confirmation:
            error = "The passwords do not match."
        else:
            try:
                with _connect_auth_db() as connection:
                    cursor = connection.execute(
                        "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                        (username, generate_password_hash(password)),
                    )
                    user_id = cursor.lastrowid
            except sqlite3.IntegrityError:
                error = "That username is already registered. Choose another one or sign in."
            else:
                session.clear()
                session.permanent = True
                session["user_id"] = user_id
                session["username"] = username
                session["csrf_token"] = secrets.token_urlsafe(32)
                return redirect(url_for("dashboard"))
    return render_template("auth.html", mode="register", error=error, username=username)


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
def dashboard():
    try:
        frame = dataset_store.load()
    except DatasetError as exc:
        return render_template("error.html", error=str(exc)), 503
    congestion = (
        frame[TARGET].astype(str).value_counts().to_dict()
        if TARGET in frame.columns
        else {}
    )
    display_congestion = {_display_label(key): int(value) for key, value in congestion.items()}
    distributions: dict[str, dict[str, int]] = {}
    for column, title in (
        ("road_type", "Road type"),
        ("traffic_intensity", "Traffic intensity"),
    ):
        if column in frame.columns:
            if column == "traffic_intensity":
                values = pd.to_numeric(frame[column], errors="coerce")
                if values.nunique(dropna=True) > 1:
                    bins = pd.qcut(values, q=4, duplicates="drop")
                    labels = {
                        str(interval): f"{interval.left:.2f}–{interval.right:.2f}"
                        for interval in bins.cat.categories
                    }
                    distributions[title] = {
                        labels[str(key)]: int(value)
                        for key, value in bins.value_counts(sort=False, dropna=True).items()
                    }
                else:
                    distributions[title] = {
                        "Single observed value": int(values.notna().sum())
                    }
            else:
                distributions[title] = {
                    str(key): int(value)
                    for key, value in frame[column].fillna("Missing").value_counts().items()
                }
    numeric_summary = frame.select_dtypes(include="number").describe().round(2)
    records = frame.head(12).fillna("").to_dict(orient="records")
    ready, model_message = _model_status(frame)
    with _prediction_lock:
        recent = list(_recent_predictions[:5])
    return render_template(
        "dashboard.html",
        frame=frame,
        row_count=len(frame),
        column_count=len(frame.columns),
        congestion=display_congestion,
        distributions=distributions,
        numeric_summary=numeric_summary,
        records=records,
        columns=frame.columns.tolist(),
        recent_predictions=recent,
        model_ready=ready,
        model_message=model_message,
    )


@app.route("/dataset")
def dataset_explorer():
    try:
        frame = dataset_store.load()
    except DatasetError as exc:
        return render_template("error.html", error=str(exc)), 503
    filtered = _filtered_data(frame)
    page_size = 25
    try:
        page = max(int(request.args.get("page", "1")), 1)
    except ValueError:
        page = 1
    page_count = max((len(filtered) + page_size - 1) // page_size, 1)
    page = min(page, page_count)
    start = (page - 1) * page_size
    visible = filtered.iloc[start : start + page_size].fillna("")
    previous_args = request.args.to_dict()
    next_args = request.args.to_dict()
    previous_args["page"] = page - 1
    next_args["page"] = page + 1
    categories = {
        column: sorted(frame[column].dropna().astype(str).unique().tolist())
        for column in ("road_type", TARGET, "district")
        if column in frame.columns
    }
    return render_template(
        "dataset.html",
        frame=frame,
        columns=frame.columns.tolist(),
        dtypes={column: str(dtype) for column, dtype in frame.dtypes.items()},
        missing=frame.isna().sum().to_dict(),
        records=visible.to_dict(orient="records"),
        filtered_count=len(filtered),
        row_count=len(frame),
        page=page,
        page_count=page_count,
        previous_url=url_for("dataset_explorer", **previous_args),
        next_url=url_for("dataset_explorer", **next_args),
        categories=categories,
        query=request.args,
    )


@app.route("/dataset/reload", methods=["POST"])
def reload_dataset():
    try:
        dataset_store.load(force=True)
    except DatasetError as exc:
        return render_template("error.html", error=str(exc)), 503
    return redirect(url_for("dashboard", reloaded="1"))


@app.route("/dataset/download")
def download_dataset():
    try:
        frame = dataset_store.load()
        filtered = _filtered_data(frame)
    except DatasetError as exc:
        return render_template("error.html", error=str(exc)), 503
    response = app.response_class(
        filtered.to_csv(index=False),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=traffic_dataset_export.csv"},
    )
    return response


@app.route("/predict", methods=["GET", "POST"])
def predict():
    try:
        frame = dataset_store.load()
        ready, message = _model_status(frame)
        if not ready:
            return render_template(
                "predict.html",
                error=message,
                schema={},
                result=None,
                recent=_recent_predictions[:5],
            ), 409
        models, metadata = _load_models()
        if metadata.get("target") != TARGET:
            raise DatasetError("The saved model target does not match congestion_level.")
    except DatasetError as exc:
        return render_template(
            "predict.html", error=str(exc), schema={}, result=None, recent=[]
        ), 503
    result = None
    error = None
    if request.method == "POST":
        payload = request.get_json(silent=True) if request.is_json else request.form.to_dict()
        payload = payload or {}
        try:
            selected = payload.get("model", "random_forest")
            if selected not in models:
                raise ValueError("Choose one of the trained models.")
            inputs = _prediction_input(payload, metadata)
            model = models[selected]
            prediction = str(model.predict(inputs)[0])
            probabilities = (
                model.predict_proba(inputs)[0]
                if hasattr(model, "predict_proba")
                else None
            )
            confidence = None
            if probabilities is not None:
                confidence = float(max(probabilities))
            result = {
                "label": prediction,
                "display_label": _display_label(prediction),
                "confidence": confidence,
                "model": selected,
            }
            with _prediction_lock:
                _recent_predictions.insert(
                    0,
                    {
                        **result,
                        "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
                del _recent_predictions[20:]
            if request.is_json:
                return jsonify(result)
        except ValueError as exc:
            if request.is_json:
                return jsonify({"error": str(exc)}), 400
            error = str(exc)
        except (TypeError, KeyError) as exc:
            error = f"Prediction could not be generated: {exc}"
            app.logger.exception("Prediction failed.")
            if request.is_json:
                return jsonify({"error": error}), 422
    with _prediction_lock:
        recent = list(_recent_predictions[:10])
    return render_template(
        "predict.html",
        error=error,
        schema=metadata["input_schema"],
        result=result,
        recent=recent,
        model_names=models.keys(),
    )


@app.route("/models")
def models_page():
    try:
        frame = dataset_store.load()
    except DatasetError as exc:
        return render_template("error.html", error=str(exc)), 503
    results_path = MODEL_DIR / "training_results.json"
    ready, message = _model_status(frame)
    metadata = None
    if results_path.is_file():
        try:
            metadata = json.loads(results_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            message = f"Could not read evaluation results: {exc}"
            ready = False
    return render_template(
        "models.html",
        ready=ready,
        message=message,
        metadata=metadata,
    )


@app.route("/models/train", methods=["POST"])
def train_models():
    try:
        train_and_save(DATASET_PATH)
        global _model_signature
        with _model_lock:
            _model_cache.clear()
            _model_signature = None
        return redirect(url_for("models_page", trained="1"))
    except TrainingError as exc:
        app.logger.error("Model training failed: %s", exc)
        return render_template(
            "error.html",
            error=f"Model training failed: {exc}",
        ), 422


@app.route("/health")
def health():
    try:
        frame = dataset_store.load()
        return jsonify({"status": "ok", "rows": len(frame), "columns": len(frame.columns)})
    except DatasetError as exc:
        return jsonify({"status": "error", "error": str(exc)}), 503


if __name__ == "__main__":
    app.run(debug=True)
