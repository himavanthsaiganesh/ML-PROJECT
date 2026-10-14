"""Train and persist traffic congestion classifiers from the project CSV."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "data" / "traffic_congestion_complete_dataset.csv"
MODEL_DIR = PROJECT_DIR / "models"
TARGET = "congestion_level"
LEAKAGE_AND_IDENTIFIER_COLUMNS = {
    "road_id",
    "road_name",
    "congestion_level",
    # This is traffic divided by capacity and is directly used to define congestion.
    "traffic_intensity",
    # This is a second representation of the maintenance date.
    "last_maintenance_date",
}
LABEL_DISPLAY = {"Low": "Low", "Medium": "Moderate", "High": "High"}


class TrainingError(RuntimeError):
    """Raised when the source data cannot be used to train the models."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_training_data(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise TrainingError(f"Dataset file was not found: {path}")
    try:
        data = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeError, ValueError) as exc:
        raise TrainingError(f"Could not read the dataset CSV: {exc}") from exc
    if data.empty:
        raise TrainingError("The dataset CSV contains no records.")
    if TARGET not in data.columns:
        raise TrainingError(f"Required target column '{TARGET}' is missing.")
    data = data.dropna(subset=[TARGET]).copy()
    if data[TARGET].nunique() < 2:
        raise TrainingError("At least two target classes are required for training.")
    if len(data) < 8:
        raise TrainingError("At least 8 labeled records are required for a train/test split.")
    return data


def _build_pipeline(
    numeric_columns: list[str], categorical_columns: list[str], estimator: Any
) -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, numeric_columns),
            ("categorical", categorical, categorical_columns),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessor", preprocessing), ("classifier", estimator)])


def _evaluation(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    predictions = model.predict(x_test)
    labels = sorted(y_test.astype(str).unique().tolist())
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, labels=labels, average="weighted", zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    classifier = model.named_steps["classifier"]
    importance: list[dict[str, Any]] = []
    if hasattr(classifier, "feature_importances_"):
        preprocessor = model.named_steps["preprocessor"]
        names = preprocessor.get_feature_names_out()
        importance = [
            {"feature": str(name), "importance": float(value)}
            for name, value in zip(names, classifier.feature_importances_)
        ]
        importance.sort(key=lambda item: item["importance"], reverse=True)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "labels": labels,
        "confusion_matrix": matrix.tolist(),
        "feature_importance": importance[:20],
    }


def train_and_save(dataset_path: Path | str = DATASET_PATH) -> dict[str, Any]:
    """Train both classifiers, evaluate them, and persist pipelines and metadata."""
    source = Path(dataset_path).resolve()
    data = _load_training_data(source)

    features = [
        column
        for column in data.columns
        if column not in LEAKAGE_AND_IDENTIFIER_COLUMNS
    ]
    if not features:
        raise TrainingError("No eligible input features remain after excluding leakage.")

    x = data.loc[:, features].copy()
    y = data[TARGET].astype(str)
    numeric_columns = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in features if column not in numeric_columns]
    class_counts = y.value_counts()
    test_count = max(int(np.ceil(len(data) * 0.25)), y.nunique())
    if test_count >= len(data) or (class_counts < 2).any():
        raise TrainingError(
            "Each congestion class needs at least two labeled records for stratified evaluation."
        )

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_count,
            random_state=42,
            stratify=y,
        )
    except ValueError as exc:
        raise TrainingError(f"Could not create a stratified train/test split: {exc}") from exc

    models: dict[str, Pipeline] = {
        "random_forest": _build_pipeline(
            numeric_columns,
            categorical_columns,
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
                min_samples_leaf=2,
            ),
        ),
        "gradient_boosting": _build_pipeline(
            numeric_columns,
            categorical_columns,
            GradientBoostingClassifier(n_estimators=100, random_state=42),
        ),
    }
    evaluations: dict[str, Any] = {}
    for name, model in models.items():
        try:
            model.fit(x_train, y_train)
        except (ValueError, TypeError) as exc:
            raise TrainingError(f"{name} training failed: {exc}") from exc
        evaluations[name] = _evaluation(model, x_test, y_test)

    source_stat = source.stat()
    metadata: dict[str, Any] = {
        "target": TARGET,
        "source_path": str(source),
        "dataset_sha256": _sha256(source),
        "dataset_modified_ns": source_stat.st_mtime_ns,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(data)),
        "columns": data.columns.tolist(),
        "feature_columns": features,
        "numeric_features": numeric_columns,
        "categorical_features": categorical_columns,
        "excluded_columns": {
            column: (
                "identifier"
                if column in {"road_id", "road_name"}
                else "target"
                if column == TARGET
                else "target leakage: derived directly from traffic and capacity"
                if column == "traffic_intensity"
                else "duplicate maintenance-date representation"
            )
            for column in data.columns
            if column not in features
        },
        "input_schema": {
            column: {
                "dtype": str(data[column].dtype),
                "kind": "numeric" if column in numeric_columns else "categorical",
                "options": (
                    []
                    if column in numeric_columns
                    else sorted(data[column].dropna().astype(str).unique().tolist())
                ),
            }
            for column in features
        },
        "split": {
            "random_state": 42,
            "test_size": int(len(x_test)),
            "train_size": int(len(x_train)),
            "stratified": True,
        },
        "class_counts": {str(key): int(value) for key, value in y.value_counts().items()},
        "evaluations": evaluations,
        "label_display": LABEL_DISPLAY,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
    temporary_metadata = MODEL_DIR / "training_results.json.tmp"
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary_metadata, MODEL_DIR / "training_results.json")
    return metadata


if __name__ == "__main__":
    try:
        result = train_and_save()
    except TrainingError as exc:
        raise SystemExit(f"Training failed: {exc}") from exc
    print(
        json.dumps(
            {
                "rows": result["rows"],
                "features": result["feature_columns"],
                "evaluations": {
                    name: {
                        key: value
                        for key, value in metrics.items()
                        if key != "feature_importance"
                    }
                    for name, metrics in result["evaluations"].items()
                },
                "saved_to": str(MODEL_DIR),
            },
            indent=2,
        )
    )
