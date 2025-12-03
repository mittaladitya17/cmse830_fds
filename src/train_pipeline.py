import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report


def load_home_credit():
    """
    Load Home Credit application_train.csv from local data folder.
    Make sure this file exists at:
    data/home-credit-default-risk/application_train.csv
    """
    csv_path = os.path.join(
        "data",
        "home-credit-default-risk",
        "application_train.csv"
    )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find {csv_path}. "
            "Download the Home Credit dataset from Kaggle and place "
            "application_train.csv in data/home-credit-default-risk/"
        )

    df = pd.read_csv(csv_path)
    return df


def make_pipeline(df: pd.DataFrame):
    """
    Build preprocessing + model pipeline for Home Credit data.
    TARGET is the label; all other columns are used as features.
    """
    # Target & features
    y = df["TARGET"]
    X = df.drop(columns=["TARGET"])

    # Optional: drop ID column if present
    if "SK_ID_CURR" in X.columns:
        X = X.drop(columns=["SK_ID_CURR"])

    # Split by type
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    # Preprocessors
    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, num_cols),
            ("cat", categorical_tf, cat_cols),
        ]
    )

    # Model – start simple with Logistic Regression
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    )

    pipe = Pipeline(steps=[
        ("pre", preprocessor),
        ("clf", model)
    ])

    metadata = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "all_feature_cols": X.columns.tolist()
    }

    return X, y, pipe, metadata


def main():
    print("🔹 Loading Home Credit data...")
    df = load_home_credit()

    # (Optional) Use a smaller sample for speed while prototyping
    if len(df) > 100_000:
        df = df.sample(100_000, random_state=42)
        print(f"Using a sample of 100,000 rows for training (from {len(df)}).")

    X, y, pipe, metadata = make_pipeline(df)

    print("🔹 Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    print("🔹 Fitting pipeline...")
    pipe.fit(X_train, y_train)

    print("🔹 Evaluating model...")
    prob = pipe.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, prob)
    print(f"ROC AUC: {auc:.4f}")
    print(classification_report(y_test, pred, digits=3))

    # Save model & metadata
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, "models/home_credit_pipeline.joblib")
    joblib.dump(metadata, "models/home_credit_metadata.joblib")
    print("✅ Saved models/home_credit_pipeline.joblib")
    print("✅ Saved models/home_credit_metadata.joblib")


if __name__ == "__main__":
    main()
