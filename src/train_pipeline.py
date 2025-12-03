# src/train_pipeline.py
# Train several models on Home Credit sample data
# and save the best pipeline + metadata.

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# try to import XGBoost, but don't crash if it's missing
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost is not installed – XGBoost model will be skipped.")

# ---------- paths ----------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent               # project root
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

DATA_PATH = DATA_DIR / "home_credit_sample.csv"
MODEL_PATH = MODELS_DIR / "credit_pipeline.joblib"
META_PATH = MODELS_DIR / "metadata.joblib"


def load_data() -> pd.DataFrame:
    """Load the Home Credit sample dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find data at: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    return df


def build_preprocessor(df: pd.DataFrame):
    """Split columns into numeric / categorical and build ColumnTransformer."""
    # target column in Home Credit dataset
    target_col = "TARGET"
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    # basic type-based split
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    # preprocessors
    cat_tf = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    num_tf = StandardScaler()

    pre = ColumnTransformer(
        transformers=[
            ("cat", cat_tf, cat_cols),
            ("num", num_tf, num_cols),
        ]
    )

    return X, y, pre, cat_cols, num_cols, target_col


def make_models():
    """Return dict of candidate models."""
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=42
        )
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        )
    return models


def train_and_evaluate():
    """Train all models, evaluate, and save the best pipeline + metadata."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_data()
    X, y, pre, cat_cols, num_cols, target_col = build_preprocessor(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    models = make_models()
    results = {}
    best_model_name = None
    best_auc = -np.inf
    best_pipe = None

    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        pipe = Pipeline([
            ("pre", pre),
            ("clf", model),
        ])

        pipe.fit(X_train, y_train)

        # predictions for metrics
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_test)[:, 1]
        else:
            # fallback: use decision_function if no proba
            if hasattr(pipe, "decision_function"):
                scores = pipe.decision_function(X_test)
                # map scores to [0,1] via sigmoid-ish transform
                proba = 1 / (1 + np.exp(-scores))
            else:
                # last fallback: use hard predictions as "probabilities"
                proba = pipe.predict(X_test)

        preds = (proba >= 0.5).astype(int)

        auc = roc_auc_score(y_test, proba)
        acc = accuracy_score(y_test, preds)

        results[name] = {
            "roc_auc": float(round(auc, 4)),
            "accuracy": float(round(acc, 4)),
        }

        print(f"{name} ROC AUC: {auc:.4f}, Accuracy: {acc:.4f}")

        # track best model by AUC
        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_pipe = pipe

    # save best model pipeline
    if best_pipe is None:
        raise RuntimeError("No model was trained successfully.")

    joblib.dump(best_pipe, MODEL_PATH)
    print(f"\nSaved best model pipeline ({best_model_name}) to {MODEL_PATH}")

    # save metadata for app.py
    metadata = {
        "best_model": best_model_name,
        "metrics": results,
        "feature_cols": X.columns.tolist(),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "target_col": target_col,
    }
    joblib.dump(metadata, META_PATH)
    print(f"Saved metadata to {META_PATH}")


if __name__ == "__main__":
    train_and_evaluate()
