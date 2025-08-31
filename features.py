import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# ----------------------------
# 1. Policy Date Features
# ----------------------------
class PolicyDateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, col="Policy Start Date", ref_date=None):
        self.col = col
        self.ref_date = pd.to_datetime(ref_date) if ref_date else pd.Timestamp.today()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.col not in X.columns:
            return pd.DataFrame(index=X.index)

        dt = pd.to_datetime(X[self.col], errors="coerce")

        policy_age = (self.ref_date - dt).dt.days

        return pd.DataFrame({
            "policy_start_year": dt.dt.year,
            "policy_start_month": dt.dt.month,
            "policy_start_dayofweek": dt.dt.dayofweek,
            "policy_start_day": dt.dt.day,
            "policy_age_days": policy_age,
            "policy_is_month_end": dt.dt.is_month_end.astype(int),
            "policy_is_quarter_end": dt.dt.is_quarter_end.astype(int),
        }, index=X.index)


# ----------------------------
# 2. Text Length Feature
# ----------------------------
class TextLength(BaseEstimator, TransformerMixin):
    def __init__(self, col="Customer Feedback"):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.col not in X.columns:
            return pd.DataFrame(index=X.index)

        lengths = X[self.col].astype(str).apply(len)
        return pd.DataFrame({f"{self.col}_length": lengths}, index=X.index)


# ----------------------------
# 3. Build Full Preprocessor
# ----------------------------
def build_preprocessor():
    numeric_features = [
        "Age", "Annual Income", "Number of Dependents",
        "Health Score", "Previous Claims", "Vehicle Age",
        "Credit Score", "Insurance Duration"
    ]

    categorical_features = [
        "Gender", "Marital Status", "Education Level",
        "Occupation", "Location", "Policy Type",
        "Smoking Status", "Exercise Frequency", "Property Type"
    ]

    # Pipelines
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Final ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("policy_date", PolicyDateFeatures("Policy Start Date"), ["Policy Start Date"]),
            ("text_length", TextLength("Customer Feedback"), ["Customer Feedback"]),
        ],
        remainder="drop"
    )

    return preprocessor