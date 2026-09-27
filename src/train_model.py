"""Train and evaluate the LUMAQ DSN 2026 machine learning models."""

from __future__ import annotations

import json
from pathlib import Path

from lumaq_ml import fit_anomaly_model, generate_energy_data, save_models, train_runtime_models


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Run the reproducible proof-of-concept training workflow."""

    data = generate_energy_data()
    data_directory = ROOT / "data"
    result_directory = ROOT / "results"
    model_directory = ROOT / "models"
    data_directory.mkdir(exist_ok=True)
    result_directory.mkdir(exist_ok=True)

    data.to_csv(data_directory / "lumaq_simulated_energy_data.csv", index=False)
    runtime_model, metrics, _ = train_runtime_models(data)
    anomaly_model, scored = fit_anomaly_model(data)
    save_models(runtime_model, anomaly_model, model_directory)

    metrics.to_csv(result_directory / "runtime_model_metrics.csv", index=False)
    summary = {
        "data_source": "simulated proof-of-concept data",
        "records": int(len(data)),
        "selected_model": str(metrics.iloc[0]["model"]),
        "mae_minutes": round(float(metrics.iloc[0]["mae_minutes"]), 3),
        "rmse_minutes": round(float(metrics.iloc[0]["rmse_minutes"]), 3),
        "r2": round(float(metrics.iloc[0]["r2"]), 4),
        "detected_anomalies": int(scored["detected_anomaly"].sum()),
    }
    (result_directory / "model_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

