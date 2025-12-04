import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# Try to import XGBoost; if not installed, we just skip that model
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# --------------------------------------------------
# Paths
# --------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

GERMAN_DATA_PATH = DATA_DIR / "credit-g.csv"
GERMAN_MODEL_PATH = MODELS_DIR / "credit_pipeline.joblib"
GERMAN_META_PATH = MODELS_DIR / "metadata.joblib"

HOME_DATA_PATH = DATA_DIR / "home_credit_sample.csv"   # your new bigger dataset sample


def require_file(p: Path, label: str):
    """Small helper to make missing paths fail loudly."""
    if not p.exists():
        raise FileNotFoundError(f"{label} not found at: {p}")
    return p


# --------------------------------------------------
# Load pre-trained German Credit pipeline (for scoring)
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_german_model():
    pipe = joblib.load(require_file(GERMAN_MODEL_PATH, "Model"))
    meta = joblib.load(require_file(GERMAN_META_PATH, "Metadata"))
    return pipe, meta


# --------------------------------------------------
# Helper for batch schema alignment
# --------------------------------------------------
def coerce_schema(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """
    Align uploaded batch data to the schema used when training the German model.
    - Keep only expected columns
    - Add any missing columns filled with NaN
    """
    expected = meta["all_cols"]
    df = df.copy()

    # Drop unexpected columns
    df = df[[c for c in df.columns if c in expected]]

    # Add missing columns
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan

    # Reorder
    df = df[expected]
    return df


# --------------------------------------------------
# Load Home Credit dataset
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def load_home_credit():
    if not HOME_DATA_PATH.exists():
        return None
    df = pd.read_csv(HOME_DATA_PATH)
    return df


# --------------------------------------------------
# Train multiple models on Home Credit sample
# --------------------------------------------------
@st.cache_resource(show_spinner=True)
def train_home_models(df: pd.DataFrame):
    """
    Train Logistic Regression, Random Forest, and (optionally) XGBoost
    on the Home Credit sample dataset.
    Handles missing values using SimpleImputer so that we don't get
    the 'all_finite' ValueError anymore.
    """
    df = df.copy()

    # Basic target / feature split
    if "TARGET" not in df.columns:
        raise ValueError("Home Credit data must contain a 'TARGET' column.")

    y = df["TARGET"]
    X = df.drop(columns=["TARGET"])

    # Drop obvious ID columns if present
    for col in ["SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV"]:
        if col in X.columns:
            X = X.drop(columns=[col])

    # Split numeric vs categorical
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Preprocessing: impute + scale numeric, impute + one-hot cat
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    models = {}

    # Logistic Regression
    pipe_lr = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    pipe_lr.fit(X_train, y_train)
    prob_lr = pipe_lr.predict_proba(X_valid)[:, 1]
    pred_lr = (prob_lr >= 0.5).astype(int)
    auc_lr = roc_auc_score(y_valid, prob_lr)
    rep_lr = classification_report(y_valid, pred_lr, digits=3)
    cm_lr = confusion_matrix(y_valid, pred_lr)

    models["Logistic Regression"] = dict(
        pipeline=pipe_lr,
        auc=auc_lr,
        report=rep_lr,
        cm=cm_lr,
    )

    # Random Forest
    pipe_rf = Pipeline(
        steps=[
            ("pre", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                    random_state=42,
                ),
            ),
        ]
    )
    pipe_rf.fit(X_train, y_train)
    prob_rf = pipe_rf.predict_proba(X_valid)[:, 1]
    pred_rf = (prob_rf >= 0.5).astype(int)
    auc_rf = roc_auc_score(y_valid, prob_rf)
    rep_rf = classification_report(y_valid, pred_rf, digits=3)
    cm_rf = confusion_matrix(y_valid, pred_rf)

    models["Random Forest"] = dict(
        pipeline=pipe_rf,
        auc=auc_rf,
        report=rep_rf,
        cm=cm_rf,
    )

    # XGBoost (optional)
    if HAS_XGB:
        pipe_xgb = Pipeline(
            steps=[
                ("pre", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=4,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        )
        pipe_xgb.fit(X_train, y_train)
        prob_xgb = pipe_xgb.predict_proba(X_valid)[:, 1]
        pred_xgb = (prob_xgb >= 0.5).astype(int)
        auc_xgb = roc_auc_score(y_valid, prob_xgb)
        rep_xgb = classification_report(y_valid, pred_xgb, digits=3)
        cm_xgb = confusion_matrix(y_valid, pred_xgb)

        models["XGBoost"] = dict(
            pipeline=pipe_xgb,
            auc=auc_xgb,
            report=rep_xgb,
            cm=cm_xgb,
        )

    results = dict(
        models=models,
        feature_cols=X.columns.tolist(),
        num_cols=num_cols,
        cat_cols=cat_cols,
        n_train=len(X_train),
        n_valid=len(X_valid),
        target_balance=y.value_counts(normalize=True).to_dict(),
    )
    return results


# --------------------------------------------------
# Streamlit layout
# --------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Analysis Dashboard",
    layout="wide",
)

st.title("Credit Risk Analysis Dashboard")

st.markdown(
    """
This app has **two main parts**:

1. **German Credit Dataset**  
   - Pre-trained **Logistic Regression** model (loaded from disk)  
   - Used for **single applicant** and **batch file** risk scoring  

2. **Home Credit Dataset (larger, real-world-like)**  
   - Trains three models right inside the app:  
     - Logistic Regression  
     - Random Forest  
     - XGBoost (if available)  
   - Used for **EDA** and **model comparison** on a bigger dataset
"""
)

# Load German model
pipe_german, meta_german = load_german_model()

# Load Home Credit sample
home_df = load_home_credit()

# Tabs
tab_overview, tab_eda, tab_metrics, tab_single, tab_batch = st.tabs(
    ["Overview", "EDA (Home Credit)", "Model Metrics", "Single Prediction", "Batch Prediction"]
)

# --------------------------------------------------
# Tab 1: Overview
# --------------------------------------------------
with tab_overview:
    st.subheader("What This App Does")

    st.markdown(
        """
**Goal:** demonstrate how data science can be used to estimate **probability of default** and
support **risk-based lending decisions**.

**Part 1 – German Credit (smaller dataset)**  
- Classic benchmark dataset  
- Clean features, simple structure  
- Pre-trained Logistic Regression model  
- Used for interactive scoring (single + batch)

**Part 2 – Home Credit (larger dataset)**  
- Realistic, high-dimensional credit data  
- Includes many socio-economic and loan-related features  
- Trains and compares multiple models:
  - Logistic Regression  
  - Random Forest  
  - XGBoost (if installed)  
- Lets us explore:
  - Class imbalance  
  - Feature types (numeric + categorical)  
  - Effect of model choice on AUC and confusion matrix
"""
    )

    if home_df is not None:
        st.markdown("**Home Credit sample loaded successfully.**")
        st.write(f"Rows: {home_df.shape[0]:,}  |  Columns: {home_df.shape[1]}")
    else:
        st.error(
            "Home Credit sample not found at `data/home_credit_sample.csv`. "
            "EDA and model comparison will be limited."
        )


# --------------------------------------------------
# Tab 2: EDA on Home Credit
# --------------------------------------------------
with tab_eda:
    st.subheader("Exploratory Data Analysis – Home Credit Sample")

    if home_df is None:
        st.warning("Home Credit dataset not available, so EDA is skipped.")
    else:
        st.markdown("##### Raw Preview")
        st.dataframe(home_df.head())

        st.markdown("##### Basic Info")
        st.write(f"Shape: {home_df.shape[0]:,} rows × {home_df.shape[1]} columns")

        if "TARGET" in home_df.columns:
            st.markdown("##### Target Distribution (`TARGET`)")
            fig, ax = plt.subplots()
            home_df["TARGET"].value_counts().sort_index().plot(kind="bar", ax=ax)
            ax.set_xlabel("TARGET (0 = non-default, 1 = default)")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        # Numeric summary & simple correlation
        num_cols_all = home_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols_all) > 1:
            st.markdown("##### Correlation Heatmap (subset of numeric features)")
            # Use a manageable subset to keep it readable
            subset = num_cols_all[:10]
            corr = home_df[subset].corr()

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)


# --------------------------------------------------
# Tab 3: Model Metrics (Home Credit)
# --------------------------------------------------
with tab_metrics:
    st.subheader("Model Performance – Home Credit Sample")

    if home_df is None:
        st.warning("Home Credit dataset not available, cannot train comparison models.")
    else:
        st.info(
            "These metrics are computed by training models on the **Home Credit sample** "
            "(with an internal train/validation split)."
        )

        try:
            results = train_home_models(home_df)
            models = results["models"]

            # AUC comparison
            auc_data = {
                "Model": [],
                "AUC": [],
            }
            for name, info in models.items():
                auc_data["Model"].append(name)
                auc_data["AUC"].append(info["auc"])

            auc_df = pd.DataFrame(auc_data).sort_values("AUC", ascending=False)
            st.markdown("##### ROC–AUC by Model")
            st.dataframe(auc_df.style.format({"AUC": "{:.3f}"}), use_container_width=True)

            fig, ax = plt.subplots()
            ax.bar(auc_df["Model"], auc_df["AUC"])
            ax.set_ylabel("ROC–AUC")
            ax.set_ylim(0.5, 1.0)
            st.pyplot(fig)

            # Detailed reports in expanders
            for name, info in models.items():
                with st.expander(f"Details: {name}"):
                    st.markdown(f"**ROC–AUC:** `{info['auc']:.3f}`")
                    st.markdown("**Classification report:**")
                    st.text(info["report"])

                    cm = info["cm"]
                    fig, ax = plt.subplots()
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=["Pred 0", "Pred 1"],
                        yticklabels=["True 0", "True 1"],
                        ax=ax,
                    )
                    ax.set_title("Confusion Matrix")
                    st.pyplot(fig)

        except Exception as e:
            st.error(
                "Could not train / evaluate models on Home Credit data. "
                f"Error: {e}"
            )


# --------------------------------------------------
# Tab 4: Single Prediction (German Credit model)
# --------------------------------------------------
with tab_single:
    st.subheader("Single Applicant Prediction (German Credit Model)")

    st.markdown(
        """
Enter applicant attributes below.  
The inputs are fed into the **pre-trained Logistic Regression model** built on the German Credit dataset.
"""
    )

    input_data = {}
    for col in meta_german["all_cols"]:
        # Simple text input for everything; the pipeline will handle types / encoding
        input_data[col] = [st.text_input(f"{col}", "")]

    if st.button("Predict Default Risk", type="primary"):
        df_input = pd.DataFrame(input_data)
        try:
            prob = pipe_german.predict_proba(df_input)[:, 1][0]
            st.metric("Estimated Default Probability", f"{prob:.2%}")
        except Exception as e:
            st.error(f"Could not score this input. Error: {e}")


# --------------------------------------------------
# Tab 5: Batch Prediction (German Credit model)
# --------------------------------------------------
with tab_batch:
    st.subheader("Batch Prediction from CSV (German Credit Model)")

    st.markdown(
        """
Upload a CSV file with **one row per applicant**.  
Columns should match the German Credit training schema; if some are missing they
will be filled with `NaN` and handled by the pipeline.
"""
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            st.markdown("##### Uploaded data (first 5 rows)")
            st.dataframe(df_up.head())

            df_aligned = coerce_schema(df_up, meta_german)
            probs = pipe_german.predict_proba(df_aligned)[:, 1]
            out = df_up.copy()
            out["default_probability"] = probs

            st.markdown("##### Scored Output (first 10 rows)")
            st.dataframe(out.head(10))

            # Allow download
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download scored CSV",
                data=csv_bytes,
                file_name="scored_applicants.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Could not process uploaded file. Error: {e}")
