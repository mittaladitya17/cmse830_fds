import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
)

# Try to import XGBoost; if not available, we just skip it
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
HERE = Path(__file__).resolve().parent          # .../src
ROOT = HERE.parent                              # repo root
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

GERMAN_PATH = DATA_DIR / "credit-g.csv"
HOME_PATH = DATA_DIR / "home_credit_sample.csv"

MODEL_PATH = MODELS_DIR / "credit_pipeline.joblib"
META_PATH = MODELS_DIR / "metadata.joblib"


def require_file(p: Path, label: str) -> Path:
    """Raise a nice error if an expected file is missing."""
    if not p.exists():
        raise FileNotFoundError(f"{label} not found at: {p}")
    return p


# -------------------------------------------------------------------
# Load saved German Credit pipeline (for predictions)
# -------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_german_model():
    pipe = joblib.load(require_file(MODEL_PATH, "Model"))
    meta = joblib.load(require_file(META_PATH, "Metadata"))
    return pipe, meta


# -------------------------------------------------------------------
# Load Home Credit sample data (for EDA + metrics)
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_home_data():
    if not HOME_PATH.exists():
        return None
    df = pd.read_csv(HOME_PATH)
    return df


# -------------------------------------------------------------------
# Train models on Home Credit sample (LogReg, RF, XGB)
# -------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_home_models(df: pd.DataFrame):
    """
    Train multiple models on the Home Credit sample.
    Returns a dict of metrics and the trained models.
    """

    target_col = "TARGET"
    if target_col not in df.columns:
        return None

    # Very simple cleaning: drop rows with missing target
    df = df.dropna(subset=[target_col])

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Identify column types
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()

    # To keep things fast, drop columns that are almost all missing
    missing_frac = X.isna().mean()
    keep_cols = missing_frac[missing_frac < 0.6].index.tolist()
    X = X[keep_cols]
    cat_cols = [c for c in cat_cols if c in keep_cols]
    num_cols = [c for c in num_cols if c in keep_cols]

    # Simple preprocessors
    cat_tf = OneHotEncoder(handle_unknown="ignore")
    num_tf = StandardScaler()

    pre = ColumnTransformer(
        transformers=[
            ("cat", cat_tf, cat_cols),
            ("num", num_tf, num_cols),
        ],
        remainder="drop",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {}
    metrics = {}

    # 1. Logistic Regression
    log_reg = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        class_weight="balanced",
    )
    pipe_lr = Pipeline([("pre", pre), ("clf", log_reg)])
    pipe_lr.fit(X_train, y_train)
    prob_lr = pipe_lr.predict_proba(X_test)[:, 1]
    pred_lr = (prob_lr >= 0.5).astype(int)
    metrics["Logistic Regression"] = {
        "Accuracy": accuracy_score(y_test, pred_lr),
        "ROC AUC": roc_auc_score(y_test, prob_lr),
    }
    models["Logistic Regression"] = pipe_lr

    # 2. Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_depth=8,
    )
    pipe_rf = Pipeline([("pre", pre), ("clf", rf)])
    pipe_rf.fit(X_train, y_train)
    prob_rf = pipe_rf.predict_proba(X_test)[:, 1]
    pred_rf = (prob_rf >= 0.5).astype(int)
    metrics["Random Forest"] = {
        "Accuracy": accuracy_score(y_test, pred_rf),
        "ROC AUC": roc_auc_score(y_test, prob_rf),
    }
    models["Random Forest"] = pipe_rf

    # 3. XGBoost (if installed)
    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        )
        pipe_xgb = Pipeline([("pre", pre), ("clf", xgb)])
        pipe_xgb.fit(X_train, y_train)
        prob_xgb = pipe_xgb.predict_proba(X_test)[:, 1]
        pred_xgb = (prob_xgb >= 0.5).astype(int)
        metrics["XGBoost"] = {
            "Accuracy": accuracy_score(y_test, pred_xgb),
            "ROC AUC": roc_auc_score(y_test, prob_xgb),
        }
        models["XGBoost"] = pipe_xgb

    return metrics, models


# -------------------------------------------------------------------
# Helper for prediction tabs (German model)
# -------------------------------------------------------------------
def coerce_schema(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Make sure uploaded / input data matches training schema."""
    cols = meta["all_cols"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]
    return df


# ===================================================================
# Streamlit UI
# ===================================================================
st.set_page_config(
    page_title="Credit Risk Analysis Dashboard",
    layout="wide",
)

st.title("Credit Risk Analysis Dashboard")

pipe_german, meta = load_german_model()
home_df = load_home_data()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "EDA (Home Credit)", "Model Metrics", "Single Prediction", "Batch Prediction"]
)

# -------------------------------------------------------------------
# Tab 1: Overview
# -------------------------------------------------------------------
with tab1:
    st.subheader("What This App Does")

    st.markdown(
        """
This dashboard is built as a **finance risk and credit scoring project**.

**Datasets:**
- 🟢 **German Credit** (UCI / OpenML): used to train the main production-style logistic regression model.  
- 🟡 **Home Credit Sample** (Kaggle, down-sampled): used for richer EDA and for training **multiple models** to compare performance.

**Models Used:**
- **Logistic Regression**
  - Interpretable baseline model.
  - Used in the **saved pipeline** for single & batch predictions.
- **Random Forest**
  - Non-linear, tree-based ensemble.
  - Handles interactions & non-linearities better.
- **XGBoost** (if available)
  - Gradient boosting model widely used in Kaggle credit-risk competitions.
  - Typically yields strong ROC AUC on tabular data.

**What you can do in this app:**
1. **Explore the Home Credit sample dataset** (EDA tab).
2. **Compare model performance** (LogReg vs Random Forest vs XGBoost) on the Home Credit sample.
3. **Score an individual applicant** using the trained German Credit pipeline.
4. **Upload a batch of applicants** and get default probabilities back as a downloadable CSV.
"""
    )

    if home_df is None:
        st.warning(
            "Note: `home_credit_sample.csv` not found in `data/`. "
            "Only the German Credit-based prediction tabs will be fully active."
        )

# -------------------------------------------------------------------
# Tab 2: EDA (Home Credit)
# -------------------------------------------------------------------
with tab2:
    st.subheader("Exploratory Data Analysis – Home Credit Sample")

    if home_df is None:
        st.error(
            "Home Credit sample data is missing. "
            "Please ensure `data/home_credit_sample.csv` is present."
        )
    else:
        st.markdown("**Dataset preview (first 5 rows):**")
        st.dataframe(home_df.head())

        st.markdown("**Basic info:**")
        st.write(f"Shape: {home_df.shape[0]} rows × {home_df.shape[1]} columns")

        if "TARGET" in home_df.columns:
            st.markdown("**Target distribution (0 = repaid, 1 = default):**")
            fig, ax = plt.subplots()
            sns.countplot(x="TARGET", data=home_df, ax=ax)
            ax.set_xlabel("TARGET")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        # Numeric summary
        st.markdown("**Summary statistics (numeric columns):**")
        st.write(home_df.select_dtypes(include="number").describe().T)

        # Simple correlation heatmap
        num_cols_small = home_df.select_dtypes(include="number").columns.tolist()
        if len(num_cols_small) > 1:
            # To avoid huge matrices, limit to first 15 numeric cols
            num_cols_small = num_cols_small[:15]
            corr = home_df[num_cols_small].corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
            ax.set_title("Correlation Heatmap (subset of numeric features)")
            st.pyplot(fig)

        # Example Plotly visualization
        if "AMT_CREDIT" in home_df.columns and "AMT_INCOME_TOTAL" in home_df.columns:
            st.markdown("**Credit vs Income (colored by TARGET when available):**")
            if "TARGET" in home_df.columns:
                fig = px.scatter(
                    home_df.sample(min(5000, len(home_df))),  # subsample for speed
                    x="AMT_INCOME_TOTAL",
                    y="AMT_CREDIT",
                    color="TARGET",
                    title="Credit Amount vs Income",
                    labels={
                        "AMT_INCOME_TOTAL": "Total Income",
                        "AMT_CREDIT": "Credit Amount",
                    },
                    opacity=0.6,
                )
            else:
                fig = px.scatter(
                    home_df.sample(min(5000, len(home_df))),
                    x="AMT_INCOME_TOTAL",
                    y="AMT_CREDIT",
                    title="Credit Amount vs Income",
                    opacity=0.6,
                )
            st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------
# Tab 3: Model Metrics (Home Credit)
# -------------------------------------------------------------------
with tab3:
    st.subheader("Model Performance Metrics (Home Credit Sample)")

    if home_df is None:
        st.error(
            "Home Credit sample data not found. Cannot compute metrics. "
            "Make sure `home_credit_sample.csv` is in the `data/` folder."
        )
    elif "TARGET" not in home_df.columns:
        st.error("Column `TARGET` not found in the Home Credit sample.")
    else:
        st.markdown(
            """
These metrics are computed by **training models on the Home Credit sample**
inside the app (logistic regression, random forest, and optionally XGBoost).

> In a “real” production system, we would train these models offline,
> save them, and evaluate on a separate validation or test set.
"""
        )

        with st.spinner("Training models on Home Credit sample..."):
            result = train_home_models(home_df)

        if result is None:
            st.error("Could not train models – check that `TARGET` exists and data is valid.")
        else:
            metrics_dict, models_dict = result

            # Display metrics in a table
            metrics_df = (
                pd.DataFrame(metrics_dict)
                .T[["Accuracy", "ROC AUC"]]
                .sort_values("ROC AUC", ascending=False)
            )
            st.markdown("### Summary Table")
            st.dataframe(metrics_df.style.format({"Accuracy": "{:.3f}", "ROC AUC": "{:.3f}"}))

            # Bar chart for ROC AUC
            st.markdown("### ROC AUC Comparison")
            plot_df = metrics_df.reset_index().rename(columns={"index": "Model"})
            fig = px.bar(
                plot_df,
                x="Model",
                y="ROC AUC",
                text=plot_df["ROC AUC"].round(3),
                title="ROC AUC by Model (Home Credit sample)",
            )
            fig.update_layout(yaxis_range=[0.5, 1.0])
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                """
**Interpretation (high-level):**

- Logistic Regression gives an interpretable baseline.
- Random Forest usually captures non-linear relationships and interactions between features.
- XGBoost (when available) often performs best on this type of tabular credit-risk data.
"""
            )

# -------------------------------------------------------------------
# Tab 4: Single Prediction (German Credit pipeline)
# -------------------------------------------------------------------
with tab4:
    st.subheader("Single Applicant Prediction (German Credit Model)")

    st.markdown(
        """
This section uses the **saved logistic regression pipeline** trained on the
German Credit dataset. It expects the original feature schema from that dataset.
"""
    )

    input_data = {}
    for col in meta["all_cols"]:
        input_data[col] = [st.text_input(f"{col}", "")]

    if st.button("Predict default risk"):
        df_input = pd.DataFrame(input_data)
        df_input = coerce_schema(df_input, meta)

        with st.spinner("Scoring applicant..."):
            prob = pipe_german.predict_proba(df_input)[:, 1][0]

        st.metric("Predicted Default Probability", f"{prob:.2%}")
        if prob > 0.5:
            st.warning("Model flags this applicant as **high risk** (probability > 50%).")
        else:
            st.success("Model flags this applicant as **lower risk** (probability ≤ 50%).")

# -------------------------------------------------------------------
# Tab 5: Batch Prediction (German Credit pipeline)
# -------------------------------------------------------------------
with tab5:
    st.subheader("Batch Prediction on CSV (German Credit Model)")

    st.markdown(
        """
Upload a CSV file with columns compatible with the **German Credit** dataset
(the same schema used to train the saved pipeline). The app will return
a CSV with an extra column containing the predicted default probability.
"""
    )

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            st.write("Uploaded data preview:")
            st.dataframe(df_up.head())

            df_up_proc = coerce_schema(df_up.copy(), meta)

            with st.spinner("Scoring all rows with the German Credit pipeline..."):
                probs = pipe_german.predict_proba(df_up_proc)[:, 1]

            df_result = df_up.copy()
            df_result["default_probability"] = probs

            st.markdown("**Sample of results:**")
            st.dataframe(df_result.head())

            # Download link
            csv_bytes = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download results as CSV",
                data=csv_bytes,
                file_name="credit_risk_predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Error while scoring file: {e}")
