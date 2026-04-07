from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "crop_yield_dataset.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "crop_yield_model.joblib"

FEATURE_COLUMNS = [
    "Temperature (°C)",
    "Rainfall (mm)",
    "Humidity (%)",
    "Soil Type",
    "Weather Condition",
    "Crop Type",
]
TARGET_COLUMN = "Yield (tons/hectare)"
NUMERIC_COLUMNS = ["Temperature (°C)", "Rainfall (mm)", "Humidity (%)"]
CATEGORICAL_COLUMNS = ["Soil Type", "Weather Condition", "Crop Type"]


def load_dataset(dataset_path: Path | str = DATASET_PATH) -> pd.DataFrame:
    return pd.read_csv(dataset_path)


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
            ("numeric", "passthrough", NUMERIC_COLUMNS),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_model(
    dataset_path: Path | str = DATASET_PATH,
    model_path: Path | str = MODEL_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, float]]:
    data = load_dataset(dataset_path)
    features = data[FEATURE_COLUMNS].copy()
    target = data[TARGET_COLUMN].copy()

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "metrics": metrics,
        },
        model_path,
    )

    return pipeline, metrics


def load_trained_model(model_path: Path | str = MODEL_PATH) -> tuple[Pipeline, dict[str, Any]]:
    payload = joblib.load(model_path)
    return payload["pipeline"], payload


def predict_yield(
    pipeline: Pipeline,
    *,
    temperature: float,
    rainfall: float,
    humidity: float,
    soil_type: str,
    weather_condition: str,
    crop_type: str,
) -> float:
    sample = pd.DataFrame(
        [
            {
                "Temperature (°C)": temperature,
                "Rainfall (mm)": rainfall,
                "Humidity (%)": humidity,
                "Soil Type": soil_type,
                "Weather Condition": weather_condition,
                "Crop Type": crop_type,
            }
        ]
    )
    return float(pipeline.predict(sample)[0])


def suggest_best_crop(
    pipeline: Pipeline,
    *,
    temperature: float,
    rainfall: float,
    humidity: float,
    soil_type: str,
    weather_condition: str,
    crop_types: list[str],
) -> dict[str, Any]:
    if not crop_types:
        raise ValueError("crop_types cannot be empty")

    samples = pd.DataFrame(
        [
            {
                "Temperature (°C)": temperature,
                "Rainfall (mm)": rainfall,
                "Humidity (%)": humidity,
                "Soil Type": soil_type,
                "Weather Condition": weather_condition,
                "Crop Type": crop,
            }
            for crop in crop_types
        ]
    )

    predictions = pipeline.predict(samples)
    ranking = [
        {"crop": crop, "predicted_yield": float(predicted_yield)}
        for crop, predicted_yield in zip(crop_types, predictions, strict=True)
    ]
    ranking.sort(key=lambda item: item["predicted_yield"], reverse=True)

    return {
        "best_crop": ranking[0]["crop"],
        "best_yield": ranking[0]["predicted_yield"],
        "ranking": ranking,
    }


if __name__ == "__main__":
    _, metrics = train_model()
    print(f"Model saved to: {MODEL_PATH}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R2: {metrics['r2']:.4f}")