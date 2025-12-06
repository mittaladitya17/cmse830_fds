import os
from pathlib import Path

import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

import joblib


# =====================================================================================
# Paths
# =====================================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


# =====================================================================================
# Utility: Load German credit (midterm dataset)
# =====================================================================================
@st.cache_data
def load_german_credit():
    path = DATA_DIR / "credit-g.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df


# =====================================================================================
# Utility: Load Home Credit *sample* (final dataset)
# =====================================================================================
@st.cache_data
def load_home_credit_sample():
    """
    Load the smaller sample of the Home Credit dataset.

    You told me you cut down the big 216 MB file and uploaded it to data/.
    I'm assuming the name is 'application_train_sample.csv'.
    Change the filename here if yours is different.
    """
    # Try a couple of common names
    candidates = [
        DATA_DIR / "application_train_sample.csv",
        DATA_DIR / "home-credit-default-risk" / "application_train_sample.csv",
        DATA_DIR / "home-credit-default-risk" / "application_train.csv",  # fallback
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            # Keep only rows where TARGET exists and is not NaN
            if "TARGET" in df.columns:
                df = df.dropna(subset=["TARGET"])
            return df
    return None


# =====================================================================================
# Load midterm pipeline (German credit) if available
# =====================================================================================
@st.cache_resource
def load_german_pipeline():
    pipe_path = MODELS_DIR / "credit_pipeline.joblib"
    meta_path = MODELS_DIR / "metadata.joblib"

    if pipe_path.exists() and meta_path.exists():
        pipe = joblib.load(pipe_path)
        meta = joblib.load(meta_path)
        return pipe, meta
    return None, None


# =====================================================================================
# Train models on Home Credit sample (LogReg, RF, XGB)
# =====================================================================================
@st.cache_resource(show_spinner="Training models on Home Credit sample…")
def train_home_models(home_df: pd.DataFrame):
    """
    Train three models on the Home Credit sample:
    - Logistic Regression
    - Random Forest
    - XGBoost (if installed)

    We:
    - Drop obvious ID columns
    - Impute missing values
    - Scale numeric features
    - One-hot encode categorical features
    - Evaluate on a held-out test set
    """
    df = home_df.copy()

    if "TARGET" not in df.columns:
        return None

    # Drop rows with missing target just in case
    df = df.dropna(subset=["TARGET"])

    # Separate target
    y = df["TARGET"].astype(int)

    # Drop obvious ID columns if present
    id_cols = [c for c in ["SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV"] if c in df.columns]
    X = df.drop(columns=["TARGET"] + id_cols)

    # Identify column types
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()

    # Basic safety: if some numeric columns are completely NaN, drop them
    num_cols = [c for c in num_cols if X[c].notna().sum() > 0]

    # Column transformer with imputation
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
        ],
        remainder="drop",
        n_jobs=None,
    )

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", n_jobs=None
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=42,
        ),
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
            tree_method="hist",
            random_state=42,
        )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    metrics_rows = []
    trained_models = {}

    for name, clf in models.items():
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("clf", clf),
            ]
        )

        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        metrics_rows.append(
            {
                "model": name,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "roc_auc": auc,
            }
        )
        trained_models[name] = pipe

    metrics_df = pd.DataFrame(metrics_rows).set_index("model").sort_values(
        "roc_auc", ascending=False
    )

    return {
        "metrics": metrics_df,
        "models": trained_models,
        "X_test": X_test,
        "y_test": y_test,
    }


# =====================================================================================
# Simple EDA helpers
# =====================================================================================
def eda_missingness(df: pd.DataFrame):
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        st.info("No missing values detected in this dataset.")
        return
    miss_pct = (miss / len(df)) * 100
    miss_df = pd.DataFrame({"column": miss.index, "missing_pct": miss_pct.values})
    fig = px.bar(
        miss_df,
        x="column",
        y="missing_pct",
        title="Percentage of Missing Values per Column",
    )
    fig.update_layout(xaxis_tickangle=-45, height=400)
    st.plotly_chart(fig, use_container_width=True)


def eda_numeric_distribution(df: pd.DataFrame, target_col: str | None = None):
    num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    if target_col and target_col in num_cols:
        num_cols.remove(target_col)
    if not num_cols:
        st.info("No numeric columns to plot.")
        return

    col = st.selectbox("Choose a numeric feature", num_cols, key="num_dist")
    if target_col and target_col in df.columns:
        fig = px.histogram(
            df,
            x=col,
            color=target_col,
            marginal="box",
            nbins=40,
            opacity=0.7,
            barmode="overlay",
            title=f"Distribution of {col} by {target_col}",
        )
    else:
        fig = px.histogram(
            df,
            x=col,
            nbins=40,
            marginal="box",
            opacity=0.7,
            title=f"Distribution of {col}",
        )
    st.plotly_chart(fig, use_container_width=True)


def eda_correlation(df: pd.DataFrame, target_col: str | None = None):
    num_df = df.select_dtypes(include=["number", "bool"])
    if num_df.empty:
        st.info("No numeric columns for correlation heatmap.")
        return
    corr = num_df.corr()
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu",
        title="Correlation Heatmap (numeric features)",
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    if target_col and target_col in num_df.columns:
        target_corr = corr[target_col].sort_values(ascending=False)
        st.markdown("**Features most correlated with TARGET:**")
        st.write(target_corr.to_frame("corr_with_target").head(10))


# =====================================================================================
# Streamlit App
# =====================================================================================
st.set_page_config(
    page_title="Credit Risk Scoring Dashboard",
    layout="wide",
    page_icon="💳",
)

st.title("💳 Credit Risk Scoring Dashboard")

st.markdown(
    """
This app is my semester-long project for **CMSE 830 – Foundations of Data Science**.  
The goal is to build and explain a **credit risk scoring pipeline** using:

- A small German credit dataset (used for the original midterm model + prediction demo)
- A larger **Home Credit** sample dataset (used for richer EDA and comparing multiple ML models)

Use the tabs below to explore the **problem, data, models, and interactive predictions**.
"""
)

# Load data & models
german_df = load_german_credit()
german_pipe, german_meta = load_german_pipeline()
home_df = load_home_credit_sample()
home_result = train_home_models(home_df) if home_df is not None else None

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📌 Project Overview",
        "📊 EDA & Data Understanding",
        "🤖 Models & Metrics (Home Credit)",
        "🧍 Single Applicant (German Credit)",
        "📂 Batch Scoring (German Credit)",
    ]
)

# -------------------------------------------------------------------------------------
# TAB 1: Overview
# -------------------------------------------------------------------------------------
with tab1:
    st.header("Project Overview")

    st.markdown(
        """
### 🧠 Problem Statement

Banks and lenders constantly need to answer the question:  
> **"If I give this person a loan, how likely are they to default?"**

This app simulates a **credit risk scoring system**:

- For individual users (single prediction)
- For groups of users (batch CSV scoring)
- With model comparison on a more realistic, higher-dimensional dataset (Home Credit sample)

---

### 📂 Datasets Used

**1. German Credit (credit-g.csv)**  
Small, classic dataset (1000 rows) with mixed categorical and numeric features.  
I use this dataset for:

- A **clean, interpretable logistic regression model**
- Interactive **single applicant** prediction
- **Batch** CSV scoring demo

**2. Home Credit Sample (application_train_sample.csv)**  
A cut-down version of the large Kaggle Home Credit dataset.  
It has:

- Many more features (dozens of socioeconomic + financial variables)
- A binary target: `TARGET` (1 = default, 0 = repaid)
- Missing values, skewed distributions, correlations — perfect for **EDA and model comparison**

---

### 🔧 Pipeline & Modeling Summary

For **Home Credit sample**, the ML pipeline does:

- **Initial Data Analysis (IDA)**  
  - Check structure: column types, target balance, missing values  
  - Drop obvious IDs (e.g., `SK_ID_CURR`) that don’t help prediction

- **Preprocessing**  
  - Numeric features: median imputation + standardization  
  - Categorical features: most frequent imputation + one-hot encoding  

- **Models Compared**
  - Logistic Regression (baseline linear model with class balancing)
  - Random Forest (non-linear ensemble, handles interactions)
  - XGBoost (if available in environment – strong gradient boosting model)

For **German Credit**, I reuse the trained logistic regression pipeline from the midterm:

- Encodes categorical variables using `OneHotEncoder`
- Scales numerical variables
- Outputs a **probability of default** for each applicant

---

### 🧭 App Navigation

- **📊 EDA & Data Understanding**  
  Explore distributions, correlations, and missing data for both datasets.

- **🤖 Models & Metrics (Home Credit)**  
  Compare Logistic Regression, Random Forest, and XGBoost on the Home Credit sample.  
  Inspect confusion matrices and full classification reports.

- **🧍 Single Applicant (German Credit)**  
  Manually enter features and get a predicted default probability.

- **📂 Batch Scoring (German Credit)**  
  Upload a CSV file and get risk scores for many applicants at once.
"""
    )

# -------------------------------------------------------------------------------------
# TAB 2: EDA
# -------------------------------------------------------------------------------------
with tab2:
    st.header("Exploratory Data Analysis")

    dataset_choice = st.radio(
        "Choose a dataset to explore:",
        ["German Credit (midterm)", "Home Credit Sample (final)"],
        horizontal=True,
    )

    if dataset_choice.startswith("German"):
        if german_df is None:
            st.error("German credit dataset not found in data/credit-g.csv.")
        else:
            st.subheader("German Credit – Snapshot")
            st.write(f"Shape: {german_df.shape[0]} rows × {german_df.shape[1]} columns")
            st.dataframe(german_df.head())

            st.subheader("Basic Statistics (numeric)")
            st.write(german_df.describe().T)

            st.subheader("Target Distribution (class)")
            if "class" in german_df.columns:
                fig = px.histogram(
                    german_df,
                    x="class",
                    title="Good vs Bad Credit (German dataset)",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Column 'class' not found in German dataset.")

            st.subheader("Missingness Pattern")
            eda_missingness(german_df)

            st.subheader("Numeric Feature Distributions")
            eda_numeric_distribution(german_df, target_col=None)

    else:
        if home_df is None:
            st.error(
                "Home Credit sample not found. "
                "Make sure application_train_sample.csv is in the data/ folder."
            )
        else:
            st.subheader("Home Credit – Snapshot")
            st.write(f"Shape: {home_df.shape[0]} rows × {home_df.shape[1]} columns")
            st.dataframe(home_df.head())

            st.subheader("Basic Statistics (numeric)")
            st.write(home_df.describe().T)

            if "TARGET" in home_df.columns:
                st.subheader("Target Distribution (TARGET)")
                fig = px.histogram(
                    home_df,
                    x="TARGET",
                    title="TARGET Distribution (0 = repaid, 1 = default)",
                )
                fig.update_xaxes(tickvals=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Missingness Pattern")
            eda_missingness(home_df)

            st.subheader("Numeric Feature Distributions")
            eda_numeric_distribution(home_df, target_col="TARGET" if "TARGET" in home_df.columns else None)

            st.subheader("Correlation Structure")
            eda_correlation(home_df, target_col="TARGET" if "TARGET" in home_df.columns else None)

# -------------------------------------------------------------------------------------
# TAB 3: Models & Metrics (Home Credit)
# -------------------------------------------------------------------------------------
with tab3:
    st.header("Model Performance – Home Credit Sample")

    if home_df is None:
        st.warning("Home Credit sample is not available, so models cannot be trained.")
    elif home_result is None:
        st.error("Something went wrong while training models on the Home Credit sample.")
    else:
        metrics_df = home_result["metrics"]

        st.markdown(
            """
These metrics are computed on a **hold-out validation set** from the Home Credit sample.  
Higher **ROC AUC** means the model better separates defaulters (TARGET = 1) from non-defaulters (TARGET = 0).
"""
        )

        st.subheader("Model Comparison Table")
        st.dataframe(metrics_df.round(3))

        # Bar plot of AUC
        fig_auc = px.bar(
            metrics_df.reset_index(),
            x="model",
            y="roc_auc",
            title="ROC AUC by Model",
            text="roc_auc",
        )
        fig_auc.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_auc.update_layout(yaxis_range=[0.5, 1.0], height=400)
        st.plotly_chart(fig_auc, use_container_width=True)

        # Choose model for deeper inspection
        model_name = st.selectbox(
            "Select a model to inspect in detail:",
            metrics_df.index.tolist(),
        )
        model = home_result["models"][model_name]
        X_test = home_result["X_test"]
        y_test = home_result["y_test"]

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        cm_df = pd.DataFrame(
            cm,
            index=["True 0 (repaid)", "True 1 (default)"],
            columns=["Pred 0", "Pred 1"],
        )

        st.subheader(f"Confusion Matrix – {model_name}")
        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale="Blues",
        )
        fig_cm.update_layout(height=400)
        st.plotly_chart(fig_cm, use_container_width=True)

        # ROC curve
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{model_name} ROC")
        )
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random",
                line=dict(dash="dash"),
            )
        )
        fig_roc.update_layout(
            title=f"ROC Curve – {model_name}",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=400,
        )
        st.plotly_chart(fig_roc, use_container_width=True)

        with st.expander("Show detailed classification report"):
            report = classification_report(y_test, y_pred, digits=3)
            st.text(report)

# -------------------------------------------------------------------------------------
# TAB 4: Single Applicant Prediction (German Credit)
# -------------------------------------------------------------------------------------
with tab4:
    st.header("Single Applicant Prediction – German Credit")

    if german_pipe is None or german_meta is None:
        st.error(
            "German credit model not found. "
            "Make sure 'credit_pipeline.joblib' and 'metadata.joblib' exist in models/."
        )
    else:
        st.markdown(
            """
Enter the features for a single German credit applicant below.  
The trained **logistic regression model** will output an estimated **probability of bad credit**.
"""
        )

        all_cols = german_meta["all_cols"]
        input_data = {}

        with st.form("single_applicant_form"):
            for col in all_cols:
                val = st.text_input(f"{col}", "")
                input_data[col] = [val]

            submitted = st.form_submit_button("Predict Risk")

        if submitted:
            df_input = pd.DataFrame(input_data)

            # Try to coerce numeric-looking columns
            for c in df_input.columns:
                try:
                    df_input[c] = pd.to_numeric(df_input[c])
                except Exception:
                    # leave as string if conversion fails
                    pass

            prob = german_pipe.predict_proba(df_input)[:, 1][0]
            st.metric("Estimated Probability of BAD credit", f"{prob:.2%}")

# -------------------------------------------------------------------------------------
# TAB 5: Batch Scoring (German Credit)
# -------------------------------------------------------------------------------------
with tab5:
    st.header("Batch Scoring – German Credit")

    if german_pipe is None or german_meta is None:
        st.error(
            "German credit model not found. "
            "Make sure 'credit_pipeline.joblib' and 'metadata.joblib' exist in models/."
        )
    else:
        st.markdown(
            """
Upload a CSV file with the **same feature columns** as the German credit dataset,  
and the model will score each row with a **probability of bad credit**.
"""
        )

        # Optional: provide a sample template if available
        sample_path = DATA_DIR / "german_batch_template.csv"
        if sample_path.exists():
            with open(sample_path, "rb") as f:
                st.download_button(
                    "Download sample batch CSV",
                    f,
                    file_name="german_batch_template.csv",
                    mime="text/csv",
                )
        else:
            st.info(
                "No sample batch file found. You can create one by exporting a few rows from credit-g.csv."
            )

        uploaded = st.file_uploader("Upload batch CSV", type=["csv"])

        if uploaded is not None:
            batch_df = pd.read_csv(uploaded)
            st.write("Preview of uploaded data:")
            st.dataframe(batch_df.head())

            # Ensure column order matches what the pipeline expects
            all_cols = german_meta["all_cols"]
            missing_cols = [c for c in all_cols if c not in batch_df.columns]

            if missing_cols:
                st.error(
                    f"The following required columns are missing in the uploaded file:\n{missing_cols}"
                )
            else:
                batch_df = batch_df[all_cols]
                probs = german_pipe.predict_proba(batch_df)[:, 1]
                result_df = batch_df.copy()
                result_df["bad_credit_probability"] = probs

                st.subheader("Scored Results (first 20 rows)")
                st.dataframe(result_df.head(20))

                # Allow download
                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download scored CSV",
                    data=csv_bytes,
                    file_name="german_batch_scored.csv",
                    mime="text/csv",
                )
