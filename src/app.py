# app.py  – Final version for CMSE 830 project
import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Try to import XGBoost – if not installed, we just skip it gracefully
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# ---------------------------------------------------------------------
# Paths and helper functions
# ---------------------------------------------------------------------
HERE = Path(__file__).resolve().parent          # .../src
ROOT = HERE.parent                              # repo root
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

GERMAN_DATA_PATH = DATA_DIR / "credit-g.csv"
GERMAN_MODEL_PATH = MODELS_DIR / "credit_pipeline.joblib"
GERMAN_META_PATH = MODELS_DIR / "metadata.joblib"

HOME_DATA_PATH = DATA_DIR / "home_credit_sample.csv"


def require_file(p: Path, label: str) -> Path:
    """Small helper to fail clearly if a file is missing."""
    if not p.exists():
        raise FileNotFoundError(f"{label} not found at: {p}")
    return p


@st.cache_resource(show_spinner=False)
def load_german_model():
    """Load the original German Credit pipeline + its metadata."""
    pipe = joblib.load(require_file(GERMAN_MODEL_PATH, "German model"))
    meta = joblib.load(require_file(GERMAN_META_PATH, "German metadata"))
    return pipe, meta


@st.cache_resource(show_spinner=False)
def load_home_data():
    """Load the Home Credit sample dataset."""
    try:
        df = pd.read_csv(require_file(HOME_DATA_PATH, "Home Credit sample"))
    except Exception as e:
        st.error(f"Could not load Home Credit sample: {e}")
        return None
    return df


@st.cache_resource(show_spinner=True)
def train_home_models(df: pd.DataFrame):
    """
    Train three models on the Home Credit sample:
      - Logistic Regression
      - Random Forest
      - XGBoost (if available)

    Returns a dict with models, metrics, feature cols, and best model name.
    """
    df = df.copy()

    # Home Credit train has TARGET as label (0 = paid, 1 = default)
    if "TARGET" not in df.columns:
        st.error("Home Credit sample must contain a 'TARGET' column.")
        return None

    y = df["TARGET"].astype(int)
    X = df.drop(columns=["TARGET"])

    # very simple missing handling (just to make sure models fit)
    # numeric -> fill 0, categorical -> "Unknown"
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    X[num_cols] = X[num_cols].fillna(0)
    X[cat_cols] = X[cat_cols].fillna("Unknown")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    models = {}
    metrics_rows = []

    # Logistic Regression
    pipe_lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=None,
    )
    pipe_lr = PipelineWrap(preprocessor, pipe_lr)
    pipe_lr.fit(X_train, y_train)
    lr_metrics = compute_metrics(pipe_lr, X_test, y_test, "Logistic Regression")
    models["Logistic Regression"] = pipe_lr
    metrics_rows.append(lr_metrics)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    pipe_rf = PipelineWrap(preprocessor, rf)
    pipe_rf.fit(X_train, y_train)
    rf_metrics = compute_metrics(pipe_rf, X_test, y_test, "Random Forest")
    models["Random Forest"] = pipe_rf
    metrics_rows.append(rf_metrics)

    # XGBoost – only if available
    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        )
        pipe_xgb = PipelineWrap(preprocessor, xgb)
        pipe_xgb.fit(X_train, y_train)
        xgb_metrics = compute_metrics(pipe_xgb, X_test, y_test, "XGBoost")
        models["XGBoost"] = pipe_xgb
        metrics_rows.append(xgb_metrics)

    metrics_df = pd.DataFrame(metrics_rows).set_index("model")

    # choose best model by ROC AUC
    best_name = metrics_df["roc_auc"].idxmax()
    return {
        "models": models,
        "metrics": metrics_df,
        "feature_cols": X.columns.tolist(),
        "best_name": best_name,
    }


class PipelineWrap:
    """
    Tiny wrapper so we can keep preprocessor + estimator together
    without using sklearn.Pipeline (to avoid worrying about sparse support
    on Streamlit Cloud, etc.).
    """

    def __init__(self, pre, est):
        self.pre = pre
        self.est = est

    def fit(self, X, y):
        Xt = self.pre.fit_transform(X)
        self.est.fit(Xt, y)
        return self

    def predict(self, X):
        Xt = self.pre.transform(X)
        return self.est.predict(Xt)

    def predict_proba(self, X):
        Xt = self.pre.transform(X)
        return self.est.predict_proba(Xt)


def compute_metrics(model, X_test, y_test, name):
    """Compute basic metrics for a binary classifier."""
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, prob),
        "f1": f1_score(y_test, pred),
    }


# ---------------------------------------------------------------------
# Streamlit layout
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Analysis Dashboard",
    layout="wide",
)

st.title("Credit Risk Analysis Dashboard")

pipe_german, meta_german = load_german_model()
home_df = load_home_data()
home_result = train_home_models(home_df) if home_df is not None else None

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "EDA", "Model Metrics", "Single Prediction", "Batch Prediction"]
)

# ---------------------------------------------------------------------
# Tab 1 – Overview
# ---------------------------------------------------------------------
with tab1:
    st.subheader("What This App Does")
    st.markdown(
        """
- Uses **two datasets**:  
  - German Credit dataset (small, classic dataset – used for quick single-applicant predictions).  
  - Home Credit sample dataset (larger, more realistic – used for EDA, model training, and batch scoring).  
- Trains three models on the Home Credit data:
  - **Logistic Regression**
  - **Random Forest**
  - **XGBoost** 
- Lets you:
  - Explore the Home Credit data interactively (EDA tab).
  - Compare model performance (Model Metrics tab).
  - Score a single German Credit applicant (Single Prediction tab).
  - Upload a CSV of Home Credit-style applicants and get risk scores (Batch Prediction tab).
        """
    )

    st.markdown("### Datasets Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**German Credit (credit-g.csv)**")
        st.write(f"Shape: `{credit_df.shape[0]} rows × {credit_df.shape[1]} columns`")
        st.dataframe(credit_df.head())

    with col2:
        if home_df is not None:
            st.markdown("**Home Credit Sample (home_credit_sample.csv)**")
            st.write(f"Shape: `{home_df.shape[0]} rows × {home_df.shape[1]} columns`")
            st.dataframe(home_df.head())
        else:
            st.info(
                "Home Credit sample not found. "
                "Make sure `home_credit_sample.csv` is in the `data/` folder."
            )

# ---------------------------------------------------------------------
# Tab 2 – EDA on Home Credit sample
# ---------------------------------------------------------------------
with tab2:
    st.subheader("Exploratory Data Analysis – Home Credit Sample")

    dataset_choice = st.radio(
        "Choose dataset to explore:",
        ["German Credit", "Home Credit Sample"],
        horizontal=True
    )

    if dataset_choice == "German Credit":
        df = credit_df.copy()
        target_col = "class"
        st.caption("Target column: `class` (good / bad).")

    if home_df is None:
        st.error("Home Credit sample could not be loaded.")
    else:
        df = home_df.copy()
        target_col = "TARGET" if "TARGET" in df.columns else None

        if target_col:
            st.caption("Target column `TARGET`: 1 = default, 0 = no default.")
        else:
            st.warning(
                "No `TARGET` column found. EDA will still work, but plots "
                "cannot be coloured by the target."
            )

        # --- Basic info cards
        st.markdown("#### Basic Info")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Rows", df.shape[0])
        with c2:
            st.metric("Columns", df.shape[1])
        with c3:
            st.metric("Missing values", int(df.isna().sum().sum()))

        # --- Summary statistics
        with st.expander("Show summary statistics"):
            try:
                st.write(df.describe(include="all", datetime_is_numeric=True).T)
            except TypeError:
                # older pandas on Streamlit Cloud may not support datetime_is_numeric
                st.write(df.describe(include="all").T)

        # identify numeric / categorical columns
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        # --- 1. Univariate distribution (numeric)
        st.markdown("### 1. Univariate Distribution (Numeric)")
        if num_cols:
            col_num = st.selectbox("Choose a numeric column:", num_cols, key="num_eda")
            color_arg = target_col if target_col else None
            fig = px.histogram(
                df,
                x=col_num,
                color=color_arg,
                marginal="box",
                nbins=40,
                title=f"Distribution of {col_num}",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns found.")

        # --- 2. Univariate distribution (categorical)
        st.markdown("### 2. Univariate Distribution (Categorical)")
        if cat_cols:
            col_cat = st.selectbox("Choose a categorical column:", cat_cols, key="cat_eda")

            if target_col:
                counts = (
                    df.groupby([col_cat, target_col])
                    .size()
                    .reset_index(name="count")
                )
                fig = px.bar(
                    counts,
                    x=col_cat,
                    y="count",
                    color=target_col,
                    barmode="group",
                    title=f"{col_cat} counts by TARGET",
                )
            else:
                counts = df[col_cat].value_counts().reset_index()
                counts.columns = [col_cat, "count"]
                fig = px.bar(
                    counts,
                    x=col_cat,
                    y="count",
                    title=f"Counts of {col_cat}",
                )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical columns found.")

        # --- 3. Correlation heatmap
        st.markdown("### 3. Correlation Heatmap (Numeric Features)")
        if len(num_cols) > 1:
            corr = df[num_cols].corr()
            fig = px.imshow(
                corr,
                text_auto=False,
                aspect="auto",
                title="Correlation between numeric features",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numeric columns for a correlation heatmap.")

        # --- 4. 2D Scatter (numeric vs numeric, optional color by target)
        st.markdown("### 4. 2D Scatter Plot")
        if len(num_cols) >= 2:
            x_sc = st.selectbox("X-axis", num_cols, index=0, key="scatter_x")
            y_sc = st.selectbox("Y-axis", num_cols, index=1, key="scatter_y")

            color_arg = target_col if target_col else None
            fig = px.scatter(
                df,
                x=x_sc,
                y=y_sc,
                color=color_arg,
                title=f"{x_sc} vs {y_sc}",
                opacity=0.6,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numeric columns for a scatter plot.")


# ---------------------------------------------------------------------
# Tab 3 – Model Metrics (Home Credit)
# ---------------------------------------------------------------------
with tab3:
    st.subheader("Model Performance Metrics (Home Credit Sample)")

    if home_df is None or home_result is None:
        st.error("Home Credit data or models are not available.")
    else:
        metrics_df = home_result["metrics"].copy()

        st.markdown(
            "These metrics are computed by training the models on the Home "
            "Credit sample and evaluating them on a held-out test split."
        )

        st.dataframe(metrics_df.style.format("{:.3f}"))

        # ROC AUC comparison bar chart
        st.markdown("#### ROC AUC Comparison")
        fig = px.bar(
            metrics_df.reset_index(),
            x="model",
            y="roc_auc",
            title="ROC AUC by Model",
        )
        st.plotly_chart(fig, use_container_width=True)

        if not HAS_XGB:
            st.info(
                "XGBoost is not installed in this environment, so only Logistic "
                "Regression and Random Forest are compared here."
            )


# ---------------------------------------------------------------------
# Tab 4 – Single prediction (German Credit model)
# ---------------------------------------------------------------------
with tab4:
    st.subheader("Single Applicant Prediction (German Credit Dataset)")

    st.markdown(
        "This form uses the **pretrained Logistic Regression pipeline** "
        "built on the German Credit dataset. It is mainly here to show how "
        "a single-applicant interface might look in production."
    )

    all_cols = meta_german["all_cols"]
    user_input = {}

    st.markdown("#### Enter Applicant Information")
    for col in all_cols:
        user_input[col] = [st.text_input(col, "")]

    if st.button("Predict Risk (German Credit)"):
        df_input = pd.DataFrame(user_input)

        try:
            prob = pipe_german.predict_proba(df_input)[:, 1][0]
            st.metric("Estimated Default Probability", f"{prob:.2%}")
            st.caption("Higher values indicate higher credit risk.")
        except Exception as e:
            st.error(f"Could not compute prediction: {e}")


# ---------------------------------------------------------------------
# Tab 5 – Batch prediction (Home Credit)
# ---------------------------------------------------------------------
with tab5:
    st.subheader("Batch Prediction – Home Credit Sample Style")

    if home_df is None or home_result is None:
        st.error(
            "Home Credit data or models are not available. "
            "Please check that `home_credit_sample.csv` exists in the data folder."
        )
    else:
        best_name = home_result["best_name"]
        best_model = home_result["models"][best_name]
        feature_cols = home_result["feature_cols"]

        st.markdown(
            f"""
We use the **best model on the Home Credit sample**:  
**{best_name}**  
to score uploaded applicants.
            """
        )

        # download template
        st.markdown("#### 1. Download CSV Template")
        template = home_df[feature_cols].head(50).copy()
        csv_bytes = template.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download batch template CSV",
            data=csv_bytes,
            file_name="home_credit_batch_template.csv",
            mime="text/csv",
        )

        st.markdown("#### 2. Upload Filled CSV")
        uploaded = st.file_uploader(
            "Upload a CSV with the same columns as the template (no TARGET column).",
            type=["csv"],
        )

        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Could not read uploaded CSV: {e}")
                batch_df = None

            if batch_df is not None:
                st.markdown("##### Preview of uploaded data")
                st.dataframe(batch_df.head())

                # align columns
                # drop extra columns, add any missing ones as NaN
                for c in batch_df.columns:
                    if c not in feature_cols:
                        st.warning(f"Dropping unexpected column: {c}")
                batch_df = batch_df[[c for c in batch_df.columns if c in feature_cols]]

                missing = [c for c in feature_cols if c not in batch_df.columns]
                for c in missing:
                    batch_df[c] = np.nan
                    st.warning(f"Adding missing column with NaN values: {c}")

                batch_df = batch_df[feature_cols]

                if st.button("Run Batch Prediction"):
                    # basic fill (same idea as training)
                    num_cols = batch_df.select_dtypes(include=["number"]).columns
                    cat_cols = batch_df.select_dtypes(exclude=["number"]).columns
                    batch_df[num_cols] = batch_df[num_cols].fillna(0)
                    batch_df[cat_cols] = batch_df[cat_cols].fillna("Unknown")

                    try:
                        probs = best_model.predict_proba(batch_df)[:, 1]
                        result = batch_df.copy()
                        result["default_probability"] = probs

                        st.markdown("##### Sample of Scored Applicants")
                        st.dataframe(result.head())

                        out_csv = result.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download results as CSV",
                            data=out_csv,
                            file_name="home_credit_batch_scored.csv",
                            mime="text/csv",
                        )
                    except Exception as e:
                        st.error(f"Could not run batch prediction: {e}")
