# src/train.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
import mlflow
import mlflow.sklearn

from src.features import build_features

TRAIN_PATH = "data/train.csv"
ARTIFACT_PATH = "artifacts/premium_model_stacked.joblib"


def main():
    print("📂 Loading data...")
    df = pd.read_csv(TRAIN_PATH)

    print("🔧 Building features...")
    preprocessor, X, y_log, _ = build_features(df)

    # Define base models
    base_models = [
        ("rf", RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)),
        ("gb", GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42))
    ]
    final_estimator = Ridge(alpha=1.0)

    # Stacking ensemble
    model = StackingRegressor(
        estimators=base_models,
        final_estimator=final_estimator,
        passthrough=True,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    print("🔹 Training Stacking Ensemble Model (Premium Prediction)...")
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y_log, cv=cv, scoring="r2", n_jobs=-1)

    print(f"Stacking CV R²: {scores.mean():.3f} ± {scores.std():.3f}")

    # Fit final pipeline
    pipe.fit(X, y_log)

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(pipe, ARTIFACT_PATH)
    print(f"✅ Stacking model trained and saved to {ARTIFACT_PATH}")

    # Log to MLflow
    with mlflow.start_run(run_name="stacking-train"):
        mlflow.log_metric("cv_r2_mean", scores.mean())
        mlflow.log_metric("cv_r2_std", scores.std())
        mlflow.sklearn.log_model(pipe, "stacking-model")
        mlflow.log_artifact(ARTIFACT_PATH)


if __name__ == "__main__":
    main()
