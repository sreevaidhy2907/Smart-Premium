import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib
from src.features import build_preprocessor  # ✅ use your new preprocessor


# ----------------------------
# Objective Function for Optuna
# ----------------------------
def objective(trial, X, y):
    # Hyperparameter search space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist"  # faster training
    }

    # Build pipeline
    preprocessor = build_preprocessor()
    model = XGBRegressor(**params)

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2", n_jobs=-1)

    return scores.mean()


# ----------------------------
# Main function
# ----------------------------
def main():
    # Load data
    df = pd.read_csv("data/train.csv")

    # Separate target
    y = np.log1p(df["Premium Amount"])  # ✅ log-transform target
    X = df.drop(columns=["Premium Amount", "id"], errors="ignore")

    # Run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=30)

    print("Best trial:")
    print(study.best_trial)

    # Train final model with best params
    best_params = study.best_trial.params
    best_params.update({"random_state": 42, "n_jobs": -1, "tree_method": "hist"})

    preprocessor = build_preprocessor()
    model = XGBRegressor(**best_params)

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    pipe.fit(X, y)

    # Save final tuned model
    joblib.dump(pipe, "artifacts/premium_model.joblib")
    print("✅ Tuned model trained and saved to artifacts/premium_model.joblib")


if __name__ == "__main__":
    main()