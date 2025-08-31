from __future__ import annotations
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from src.features import DateFeatures, TextLength
from src.schema import schema

SEED = 42


def build_pipeline(df: pd.DataFrame) -> Pipeline:
    target_col = schema["target"]
    X = df.drop(columns=[target_col])

    num_cols = [c for c in schema["numeric"] if c in X.columns]
    cat_cols = [c for c in schema["categorical"] if c in X.columns]
    text_cols = [c for c in schema["text"] if c in X.columns]
    date_cols = [c for c in schema["datetime"] if c in X.columns]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    steps = []
    if date_cols:
        steps.append(("date", DateFeatures(date_cols[0])))
    if text_cols:
        steps.append(("text", TextLength(text_cols[0])))

    steps.extend([
        ("pre", pre),
        ("model", XGBRegressor(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.06,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            tree_method="hist",
        )),
    ])

    return Pipeline(steps)


def main() -> None:
    df = pd.read_csv("data/train.csv")
    target_col = schema["target"]
    pipe = build_pipeline(df)

    X = df.drop(columns=[target_col])
    y = np.log1p(df[target_col])

    # Cross-validation with multiple metrics
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    }
    cv_res = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    mae_mean, mae_std = -cv_res["test_MAE"].mean(), cv_res["test_MAE"].std()
    rmse_mean, rmse_std = -cv_res["test_RMSE"].mean(), cv_res["test_RMSE"].std()
    r2_mean, r2_std = cv_res["test_R2"].mean(), cv_res["test_R2"].std()

    print(f"CV MAE : {mae_mean:.3f} ± {mae_std:.3f}")
    print(f"CV RMSE: {rmse_mean:.3f} ± {rmse_std:.3f}")
    print(f"CV R2  : {r2_mean:.3f} ± {r2_std:.3f}")

    # Final fit
    pipe.fit(X, y)
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(pipe, "artifacts/premium_model.joblib")
    print("✅ Model trained and saved to artifacts/premium_model.joblib")


if __name__ == "__main__":
    main()