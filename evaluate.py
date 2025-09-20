# src/evaluate.py
import os
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import mlflow

from src.features import build_features

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
ARTIFACT_PATH = "artifacts/premium_model_stacked.joblib"
OUT_SUB_PATH = "submission.csv"


def evaluate_predictions(y_true, y_pred_log, label=""):
    """Evaluate metrics in log and original premium scale."""
    # log scale
    r2_log = r2_score(y_true, y_pred_log)
    mae_log = mean_absolute_error(y_true, y_pred_log)
    rmse_log = mean_squared_error(y_true, y_pred_log) ** 0.5

    # original scale
    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred_log)

    r2_orig = r2_score(y_true_orig, y_pred_orig)
    mae_orig = mean_absolute_error(y_true_orig, y_pred_orig)
    rmse_orig = mean_squared_error(y_true_orig, y_pred_orig) ** 0.5

    print(f"\n=== Evaluation on {label} Data ===")
    print(f"Log Scale -> R²: {r2_log:.4f}, MAE: {mae_log:.4f}, RMSE: {rmse_log:.4f}")
    print(f"Original Premium Scale -> R²: {r2_orig:.4f}, MAE: {mae_orig:.2f}, RMSE: {rmse_orig:.2f}")

    return {
        "r2_log": r2_log, "mae_log": mae_log, "rmse_log": rmse_log,
        "r2_orig": r2_orig, "mae_orig": mae_orig, "rmse_orig": rmse_orig
    }


def plot_predictions(y_true_log, y_pred_log, out_path="artifacts/pred_vs_actual.png"):
    """Scatter plot Predicted vs Actual Premiums."""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.2, s=10)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
             color="red", linestyle="--")
    plt.xlabel("Actual Premium")
    plt.ylabel("Predicted Premium")
    plt.title("Predicted vs Actual Premiums")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def feature_importance_plot(model, out_path="artifacts/feature_importance.png"):
    """Save feature importance if available."""
    try:
        final_model = model.named_steps["regressor"].final_estimator_
        if hasattr(final_model, "coef_"):
            importances = final_model.coef_
            features = model.named_steps["preprocessor"].get_feature_names_out()
            imp_df = pd.DataFrame({"feature": features, "importance": importances})
            imp_df.sort_values("importance", ascending=False).head(20).plot(
                x="feature", y="importance", kind="bar", figsize=(8, 5)
            )
            plt.title("Top Feature Importances (Ridge Final Estimator)")
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
            return out_path
    except Exception:
        return None
    return None


def evaluate_on_train():
    print("📂 Loading train data...")
    df = pd.read_csv(TRAIN_PATH)
    preprocessor, X, y_log, _ = build_features(df)

    print("📦 Loading model...")
    model = joblib.load(ARTIFACT_PATH)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    preds_log = model.predict(X_val)

    metrics = evaluate_predictions(y_val, preds_log, label="Train/Validation")
    plot_path = plot_predictions(y_val, preds_log)
    fi_path = feature_importance_plot(model)

    with mlflow.start_run(run_name="evaluate-train"):
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        mlflow.log_artifact(plot_path, artifact_path="plots")
        if fi_path:
            mlflow.log_artifact(fi_path, artifact_path="plots")


def predict_on_test():
    print("📂 Loading test data...")
    df_test = pd.read_csv(TEST_PATH)

    print("📦 Loading model...")
    model = joblib.load(ARTIFACT_PATH)

    preprocessor, X_test, _, _ = build_features(df_test)
    preds_log = model.predict(X_test)
    preds_orig = np.expm1(preds_log)

    sub = pd.DataFrame({
        "ID": df_test["ID"] if "ID" in df_test.columns else np.arange(len(df_test)),
        "Premium Amount": preds_orig
    })
    sub.to_csv(OUT_SUB_PATH, index=False)
    print(f"✅ Predictions saved to {OUT_SUB_PATH}")

    with mlflow.start_run(run_name="evaluate-test"):
        mlflow.log_metric("pred_min", float(preds_orig.min()))
        mlflow.log_metric("pred_max", float(preds_orig.max()))
        mlflow.log_metric("pred_mean", float(preds_orig.mean()))
        mlflow.log_artifact(OUT_SUB_PATH, artifact_path="submissions")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    args = parser.parse_args()

    if args.mode == "train":
        evaluate_on_train()
    elif args.mode == "test":
        predict_on_test()


if __name__ == "__main__":
    main()
