# Playground Series S6E2 — Predicting Heart Disease

Binary classification: predict Heart Disease (Presence/Absence) from 13 clinical features.

## Competition

- **Kaggle**: https://www.kaggle.com/competitions/playground-series-s6e2
- **Metric**: AUC
- **Train**: 630k rows, **Test**: 270k rows

## Original Data

The competition's synthetic data is generated from the original UCI heart disease dataset:

- **Source**: https://www.kaggle.com/datasets/neurocipher/heartdisease/data?select=Heart_Disease_Prediction.csv
- **Rows**: 270
- **License**: Apache 2.0

The original data is combined with the synthetic training data during training to provide additional real-world signal.

## Scripts

- `train_s6e2_baseline.py` — XGBoost + LightGBM + CatBoost ensemble with 10-fold CV
- `submit_s6e2.sh` — Submit predictions via Kaggle CLI
- `explore_s6e2.py` — EDA and data exploration
