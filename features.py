# src/features.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def build_features(df: pd.DataFrame):
    """
    Build numeric and categorical features, clip outliers,
    and prepare preprocessor pipeline.
    """

    # ---- Clip target (Premium Amount) ----
    y = None
    if "Premium Amount" in df.columns:
        y = df["Premium Amount"].clip(lower=100, upper=1300)

    # ---- Numeric features ----
    df["income_x_credit"] = df["Annual Income"] * df["Credit Score"]
    df["age_x_health"] = df["Age"] * df["Health Score"]
    df["income_per_age"] = df["Annual Income"] / (df["Age"] + 1)
    df["claims_per_income"] = df["Previous Claims"] / (df["Annual Income"] + 1)
    df["log_income"] = np.log1p(df["Annual Income"])
    df["log_claims"] = np.log1p(df["Previous Claims"])

    numeric_features = [
        "Age",
        "Annual Income",
        "Health Score",
        "Credit Score",
        "Previous Claims",
        "Vehicle Age",
        "income_x_credit",
        "age_x_health",
        "income_per_age",
        "claims_per_income",
        "log_income",
        "log_claims",
    ]

    # ---- Categorical features ----
    categorical_features = ["Policy Type", "Property Type", "Occupation", "Location"]

    # ---- Preprocessor ----
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # ---- Outputs ----
    y = None
    y_log = None
    y_bins = None
    if "Premium Amount" in df.columns:
     y = df["Premium Amount"].clip(lower=100, upper=1300)
     y_log = np.log1p(y)  # log target
     y_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")  # classification bins

    return preprocessor, df[numeric_features + categorical_features], y_log, y_bins
