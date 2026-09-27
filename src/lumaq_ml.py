"""Core data and modelling functions for the LUMAQ DSN 2026 prototype."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
FEATURES = [
    "battery_voltage_v",
    "battery_current_a",
    "state_of_charge_pct",
    "load_power_w",
    "solar_input_w",
    "ambient_temperature_c",
    "hour",
    "daylight",
    "load_category",
]
TARGET = "runtime_minutes"


@dataclass(frozen=True)
class SystemProfile:
    """Electrical assumptions used by the reproducible demonstration dataset."""

    nominal_voltage_v: float = 48.0
    battery_capacity_ah: float = 100.0
    usable_depth_of_discharge: float = 0.80
    inverter_efficiency: float = 0.90


def generate_energy_data(
    rows: int = 2400,
    seed: int = RANDOM_STATE,
    profile: SystemProfile = SystemProfile(),
) -> pd.DataFrame:
    """Generate transparent proof-of-concept data for a 48 V solar installation.

    The returned records are simulated and do not represent field measurements.
    The calculation preserves relationships between state of charge, load,
    charging input, conversion losses and remaining runtime.
    """

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-08-01", periods=rows, freq="5min")
    hours = timestamps.hour + timestamps.minute / 60
    daylight = ((hours >= 6.0) & (hours <= 18.5)).astype(int)

    solar_curve = np.maximum(0, np.sin((hours - 6) / 12.5 * np.pi))
    cloud_factor = rng.beta(5, 2, rows)
    solar_input = daylight * solar_curve * cloud_factor * rng.uniform(500, 1800, rows)

    categories = np.array(["essential", "productive", "comfort", "heavy"])
    load_category = rng.choice(categories, rows, p=[0.34, 0.28, 0.25, 0.13])
    load_ranges = {
        "essential": (80, 350),
        "productive": (180, 750),
        "comfort": (250, 1100),
        "heavy": (900, 2600),
    }
    load_power = np.array([rng.uniform(*load_ranges[value]) for value in load_category])
    evening_multiplier = np.where((hours >= 18) | (hours < 1), rng.uniform(1.05, 1.35, rows), 1.0)
    load_power *= evening_multiplier

    state_of_charge = rng.uniform(18, 100, rows)
    battery_voltage = 44.5 + (state_of_charge / 100) * 9.2 + rng.normal(0, 0.22, rows)
    net_load = np.maximum(load_power - solar_input, 25)
    battery_current = net_load / np.maximum(battery_voltage * profile.inverter_efficiency, 1)

    stored_wh = (
        profile.nominal_voltage_v
        * profile.battery_capacity_ah
        * profile.usable_depth_of_discharge
        * profile.inverter_efficiency
        * state_of_charge
        / 100
    )
    runtime_minutes = np.clip((stored_wh / net_load) * 60, 5, 1440)
    runtime_minutes *= rng.normal(1.0, 0.035, rows)

    ambient_temperature = 25 + 7 * solar_curve + rng.normal(0, 1.8, rows)

    anomaly = np.zeros(rows, dtype=int)
    anomaly_indices = rng.choice(rows, size=max(1, rows // 25), replace=False)
    load_power[anomaly_indices] *= rng.uniform(1.6, 2.3, len(anomaly_indices))
    anomaly[anomaly_indices] = 1

    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "battery_voltage_v": battery_voltage,
            "battery_current_a": battery_current,
            "state_of_charge_pct": state_of_charge,
            "load_power_w": load_power,
            "solar_input_w": solar_input,
            "ambient_temperature_c": ambient_temperature,
            "hour": hours,
            "daylight": daylight,
            "load_category": load_category,
            "runtime_minutes": runtime_minutes,
            "known_anomaly": anomaly,
            "data_source": "simulated",
        }
    )
    return data.round(
        {
            "battery_voltage_v": 2,
            "battery_current_a": 2,
            "state_of_charge_pct": 2,
            "load_power_w": 2,
            "solar_input_w": 2,
            "ambient_temperature_c": 2,
            "hour": 2,
            "runtime_minutes": 2,
        }
    )


def build_runtime_pipeline(model_name: str = "gradient_boosting") -> Pipeline:
    """Create a preprocessing and runtime-regression pipeline."""

    numeric_features = [column for column in FEATURES if column != "load_category"]
    categorical_features = ["load_category"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )
    models = {
        "gradient_boosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE, n_estimators=180, learning_rate=0.05, max_depth=3
        ),
        "random_forest": RandomForestRegressor(
            random_state=RANDOM_STATE, n_estimators=180, min_samples_leaf=2, n_jobs=-1
        ),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("preprocessor", preprocessor), ("model", models[model_name])])


def evaluate_regression(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    """Calculate regression metrics used in the project report."""

    return {
        "mae_minutes": float(mean_absolute_error(y_true, y_pred)),
        "rmse_minutes": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_runtime_models(data: pd.DataFrame) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    """Train two candidate models and return the best pipeline and predictions."""

    train, test = train_test_split(data, test_size=0.20, random_state=RANDOM_STATE)
    rows = []
    predictions = test[["timestamp", TARGET]].copy()
    fitted = {}
    for model_name in ("gradient_boosting", "random_forest"):
        pipeline = build_runtime_pipeline(model_name)
        pipeline.fit(train[FEATURES], train[TARGET])
        prediction = pipeline.predict(test[FEATURES])
        metrics = evaluate_regression(test[TARGET], prediction)
        rows.append({"model": model_name, **metrics})
        predictions[f"prediction_{model_name}"] = prediction
        fitted[model_name] = pipeline

    metrics_frame = pd.DataFrame(rows).sort_values("mae_minutes").reset_index(drop=True)
    best_name = metrics_frame.loc[0, "model"]
    return fitted[best_name], metrics_frame, predictions


def fit_anomaly_model(data: pd.DataFrame) -> Tuple[IsolationForest, pd.DataFrame]:
    """Fit an Isolation Forest and attach anomaly scores to a copy of the data."""

    columns = ["battery_voltage_v", "battery_current_a", "load_power_w", "solar_input_w"]
    model = IsolationForest(contamination=0.04, random_state=RANDOM_STATE)
    output = data.copy()
    output["anomaly_score"] = model.fit_predict(output[columns])
    output["detected_anomaly"] = (output["anomaly_score"] == -1).astype(int)
    return model, output


def energy_guidance(record: Dict[str, float]) -> str:
    """Convert a prediction record into a direct energy-use message."""

    runtime = float(record["predicted_runtime_minutes"])
    load = float(record["load_power_w"])
    state_of_charge = float(record["state_of_charge_pct"])
    anomaly = int(record.get("detected_anomaly", 0))

    if anomaly:
        return "LUMAQ detected an unusual consumption pattern. The connected loads require inspection."
    if state_of_charge < 25 or runtime < 45:
        return f"Approximately {runtime:.0f} minutes remain. Heavy nonessential loads should be disconnected."
    if load > 1200:
        return f"The present load is {load:.0f} W. Reducing heavy loads will extend the available runtime."
    return f"Approximately {runtime:.0f} minutes remain at the present consumption rate."


def save_models(runtime_model: Pipeline, anomaly_model: IsolationForest, directory: Path) -> None:
    """Persist fitted models for the dashboard."""

    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(runtime_model, directory / "runtime_model.joblib")
    joblib.dump(anomaly_model, directory / "anomaly_model.joblib")
