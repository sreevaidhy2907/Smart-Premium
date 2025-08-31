from __future__ import annotations
import joblib
import pandas as pd
import numpy as np

TARGET = "Premium Amount"

def main() -> None:
    model = joblib.load("artifacts/premium_model.joblib")

    test = pd.read_csv("data/test.csv")
    sample = pd.read_csv("data/sample_submission.csv")

    X = test.drop(columns=[TARGET], errors="ignore")

    # Predictions in log-space → back-transform
    log_preds = model.predict(X)
    preds = np.expm1(log_preds)

    # Align to sample_submission format
    sub = sample.copy()
    if TARGET in sub.columns:
        sub[TARGET] = preds
    else:
        # If target name differs, put preds in the last column
        last_col = sub.columns[-1]
        sub[last_col] = preds

    sub.to_csv("data/submission.csv", index=False)
    print("Predictions saved to data/submission.csv")

if __name__ == "__main__":
    main()