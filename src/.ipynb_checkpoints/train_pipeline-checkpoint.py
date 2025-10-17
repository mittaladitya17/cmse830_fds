# --- train_pipeline.py ---
import os
import joblib
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

# 1. Load the dataset
def load_data():
    try:
        credit = fetch_openml("credit-g", version=1, as_frame=True)
        df = credit.frame.copy()
    except Exception:
        df = pd.read_csv(os.path.join("data", "credit-g.csv"))
    return df

# 2. Build pipeline: preprocessing + logistic regression
def make_pipeline(df: pd.DataFrame):
    y = (df["class"] == "bad").astype(int)
    X = df.drop(columns=["class"])

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include="number").columns.tolist()

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", StandardScaler(), num_cols)
    ])

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")

    pipe = Pipeline([
        ("pre", pre),
        ("clf", clf)
    ])
    return X, y, pipe, cat_cols, num_cols

# 3. Train and save the model
def main():
    df = load_data()
    X, y, pipe, cat_cols, num_cols = make_pipeline(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipe.fit(X_train, y_train)

    prob = pipe.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    print("ROC AUC:", roc_auc_score(y_test, prob).round(4))
    print(classification_report(y_test, pred, digits=3))

    os.makedirs("../models", exist_ok=True)
    joblib.dump(pipe, "../models/credit_pipeline.joblib")
    joblib.dump({
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "all_cols": X.columns.tolist()
    }, "../models/metadata.joblib")
    print("✅ Model and metadata saved!")

if __name__ == "__main__":
    main()
