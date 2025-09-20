# src/predict.py
import os
import joblib
import pandas as pd
import numpy as np

from src.features import build_features  # ✅ import feature builder

TEST_PATH = "data/test.csv"
SAMPLE_SUB_PATH = "data/sample_submission.csv"
OUT_PATH = "submission.csv"


def main(model_name="stacked"):
    model_path = f"artifacts/premium_model_{model_name}.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model missing at {model_path}. Run `python -m src.train {model_name}` first.")
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError("Test file data/test.csv not found.")

    # Load model and test data
    model = joblib.load(model_path)
    df_test = pd.read_csv(TEST_PATH)

    # Apply same feature engineering
    preprocessor, X_test, _, _ = build_features(df_test)

    # Predict in log scale → invert → clip
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)
    preds = np.clip(preds, 100, 1300)

    # Save predictions in required format
    if os.path.exists(SAMPLE_SUB_PATH):
        sample = pd.read_csv(SAMPLE_SUB_PATH)
        if "Premium Amount" in sample.columns:
            sample["Premium Amount"] = preds
            sample.to_csv(OUT_PATH, index=False)
        else:
            col = sample.columns[1] if len(sample.columns) > 1 else "Premium Amount"
            sample[col] = preds
            sample.to_csv(OUT_PATH, index=False)
    else:
        if "id" in df_test.columns:
            out = pd.DataFrame({"id": df_test["id"], "Premium Amount": preds})
        else:
            out = pd.DataFrame({"Premium Amount": preds})
        out.to_csv(OUT_PATH, index=False)

    print(f"✅ Predictions saved to {OUT_PATH}")

    # Sanity check
    print("\n📊 Predictions summary:")
    print(f"Min:  {preds.min():.2f}")
    print(f"Max:  {preds.max():.2f}")
    print(f"Mean: {preds.mean():.2f}")


if __name__ == "__main__":
    import sys
    model_arg = sys.argv[1] if len(sys.argv) > 1 else "stacked"
    main(model_arg)
