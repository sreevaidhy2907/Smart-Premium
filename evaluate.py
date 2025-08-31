from __future__ import annotations
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET = "Premium Amount"

def main():
    df = pd.read_csv("data/test.csv")
    model = joblib.load("artifacts/premium_model.joblib")

    if TARGET not in df.columns:
        print("⚠️ Test set has no target column. Only predictions will be generated.")
        preds = np.expm1(model.predict(df))
        df["Predicted"] = preds
        df.to_csv("data/test_with_predictions.csv", index=False)
        return

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    log_preds = model.predict(X)
    preds = np.expm1(log_preds)

    mae = mean_absolute_error(y, preds)
    rmse = mean_squared_error(y, preds, squared=False)
    r2 = r2_score(y, preds)
    mape = np.mean(np.abs((y - preds) / y)) * 100

    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²  : {r2:.3f}")
    print(f"MAPE: {mape:.2f}%")

    df["Predicted"] = preds
    df.to_csv("data/test_with_predictions.csv", index=False)
    print("✅ Saved predictions to data/test_with_predictions.csv")


if __name__ == "__main__":
    main()