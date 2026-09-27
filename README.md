# LUMAQ DSN 2026

LUMAQ is an intelligent energy guidance system developed to help solar-powered Nigerian homes and small businesses understand how long stored energy will last and how connected loads affect available runtime.

This repository contains the machine learning proof of concept submitted for the Data Science Nigeria 2026 AI Bootcamp Hackathon under the Machine Learning with Python track.

## Problem

Solar and inverter users often see voltage, current or battery percentage without receiving direct guidance about remaining runtime. An unexpected high-power load can discharge a battery before the user recognises the change. This uncertainty affects studying, productive work and business continuity during grid outages.

## Project origin

The project emerged from Ayoola Timilehin Israel's experience as a mechatronics engineering student in Minna. Grid failures and early backup-power shutdowns interrupted study and technical work. Practical exposure to embedded systems and 48 V DC power systems showed that many users invest in solar equipment while continuing to manage stored energy through guesswork.

## System

The LUMAQ hardware prototype provides real-time energy metering, offline operation and a live monitoring interface. The machine learning layer in this repository performs two tasks:

1. Remaining-runtime prediction from battery, load and solar conditions
2. Abnormal-consumption detection from multivariate energy readings

The output layer converts predictions into direct messages that state the estimated remaining time and identify conditions requiring load inspection.

## Data status

The current notebook uses a reproducible simulated dataset based on a 48 V solar battery profile. The supplied project materials did not contain exported sensor readings. Every generated row therefore carries the value `simulated` in the `data_source` field.

The simulated dataset supports verification of the Python workflow. It does not represent field performance or replace validation with the physical gateway.

## Machine learning workflow

The notebook performs the following operations:

1. Generates and validates the demonstration dataset
2. Examines energy-use patterns and target relationships
3. Creates model-ready numeric and categorical features
4. Compares Gradient Boosting and Random Forest regression
5. Evaluates test predictions with MAE, RMSE and R²
6. Fits an Isolation Forest for abnormal-consumption detection
7. Produces an energy guidance message from a sample operating condition
8. Saves reusable model artefacts

## Repository structure

```text
LUMAQ-DSN-2026/
├── app/streamlit_app.py
├── data/data_dictionary.md
├── hardware/hardware_profile.md
├── models/model_information.md
├── notebooks/LUMAQ_DSN_2026.ipynb
├── results/model_summary.json
├── src/lumaq_ml.py
├── src/train_model.py
├── LICENSE
├── README.md
└── requirements.txt
```

## Execution

The notebook opens directly in Google Colab and contains the complete DSN demonstration. A local environment can execute the repository with:

```bash
pip install -r requirements.txt
python src/train_model.py
streamlit run app/streamlit_app.py
```

## Current scope

The repository demonstrates the software pipeline and its integration contract. The machine learning results reflect simulated records. Firmware, calibrated sensor logs and deployment measurements remain outside the submitted software package because those files were not included in the available project materials.

## Author

Ayoola Timilehin Israel  
Mechatronics Engineering, Federal University of Technology, Minna  
[Kaggle](https://www.kaggle.com/ayoolatimilehin)

## Team context

The wider LUMAQ prototype was developed with Aishat Adebanjo and Omoniyi Victor at the Federal University of Technology, Minna. Ayoola Timilehin Israel contributed software development, firmware development, embedded systems programming and technical documentation.

