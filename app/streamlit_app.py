"""Interactive LUMAQ runtime prediction demonstration."""

from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lumaq_ml import FEATURES, energy_guidance  # noqa: E402


st.set_page_config(page_title="LUMAQ Energy Guidance", page_icon="⚡", layout="wide")
st.title("LUMAQ Energy Guidance")
st.caption("DSN 2026 machine learning proof of concept")

model_path = ROOT / "models" / "runtime_model.joblib"
if not model_path.exists():
    st.error("The runtime model has not been generated. Run python src/train_model.py first.")
    st.stop()

model = joblib.load(model_path)

left, right = st.columns(2)
with left:
    voltage = st.number_input("Battery voltage (V)", 40.0, 58.0, 50.4, 0.1)
    current = st.number_input("Battery current (A)", 0.0, 100.0, 12.0, 0.5)
    state_of_charge = st.slider("State of charge (%)", 5, 100, 65)
    load_power = st.number_input("Connected load (W)", 25.0, 5000.0, 620.0, 25.0)
with right:
    solar_input = st.number_input("Solar input (W)", 0.0, 5000.0, 350.0, 25.0)
    temperature = st.number_input("Ambient temperature (°C)", 10.0, 55.0, 30.0, 0.5)
    hour = st.slider("Hour", 0.0, 23.75, 14.0, 0.25)
    category = st.selectbox("Load category", ["essential", "productive", "comfort", "heavy"])

record = pd.DataFrame(
    [
        {
            "battery_voltage_v": voltage,
            "battery_current_a": current,
            "state_of_charge_pct": state_of_charge,
            "load_power_w": load_power,
            "solar_input_w": solar_input,
            "ambient_temperature_c": temperature,
            "hour": hour,
            "daylight": int(6 <= hour <= 18.5),
            "load_category": category,
        }
    ]
)

if st.button("Estimate runtime", type="primary"):
    prediction = max(0.0, float(model.predict(record[FEATURES])[0]))
    guidance_record = record.iloc[0].to_dict()
    guidance_record["predicted_runtime_minutes"] = prediction
    st.metric("Predicted remaining runtime", f"{prediction:.0f} minutes")
    st.info(energy_guidance(guidance_record))
    st.caption("The displayed model uses simulated proof-of-concept data and requires field validation.")

