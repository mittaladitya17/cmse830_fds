# src/app.py
import os
from pathlib import Path

import numpy as np
import pandas as pd

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import joblib
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Try to import XGBoost, but don't crash app if not available
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


# --------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent           # repo root
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

GERMAN_PATH = DATA_DIR / "credit-g.csv"
GERMAN_MODEL_PATH = MODELS_DIR / "credit_pipeline.joblib"
GERMAN_META_PATH = MODELS_DIR / "metadata.joblib"

HOME_SAMPLE_PATH = DATA_DIR / "home_credit_sample.csv"


# --------------------------------------------------------------------
# SMALL UTIL
# --------------------------------------------------------------------
def require_file(p: Path, label: str) -> Path:
    """Raise a clean error if expected file is missing."""
    if not p.exists():
        raise FileNotFoundError(f"{label} not found at: {p}")
    return p


# --------------------------------------------------------------------
# LOAD GERMAN CREDIT: PRETRAINED PIPELINE (MIDTERM PART)
# --------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_german_model():
    """Load pre-trained German Credit pipeline + metadata."""
    pipe = joblib.load(require_file(GERMAN_MODEL_PATH, "German model"))
    meta = joblib.load(require_file(GERMAN_META_PATH, "German metadata"))
    return pipe, meta


@st.cache_data(show_spinner=False)
def load_german_data():
    df = pd.read_csv(require_file(GERMAN_PATH, "credit-g.csv"))
    return df


# --------------------------------------------------------------------
# LOAD HOME CREDIT SAMPLE
# --------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_home_credit_sample():
    """Load the smaller Home Credit training sample from data/."""
    if not HOME_SAMPLE_PATH.exists():
        st.warning(
            f"Home Credit sample file not found at {HOME_SAMPLE_PATH}. "
            "EDA and advanced models will be limited."
        )
        return None
    df = pd.read_csv(HOME_SAMPLE_PATH)
    return df


# --------------------------------------------------------------------
# TRAIN HOME CREDIT MODELS (LOGREG + RF + XGB)
# --------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def train_home_models(df: pd.DataFrame):
    """
    Train three models on the Home Credit sample dataset:
    - Logistic Regression
    - Random Forest
    - XGBoost (if available)

    Returns a dictionary containing:
    - models: dict[model_name -> trained pipeline]
    - summary: DataFrame with ROC AUC, Accuracy, Precision, Recall, F1
    - best_name: name of best model by ROC AUC
    - best_model: the trained pipeline of the best model
    - X_test, y_test: validation split used for evaluation
    """
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        roc_auc_score,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )

    # Try to import XGBoost
    try:
        from xgboost import XGBClassifier
        HAS_XGB = True
    except Exception:
        HAS_XGB = False

    # ---- 1. Basic cleaning & target split ----
    target_col = "TARGET"
    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' not found in dataset.")
        return None

    # Drop obvious ID columns if they exist
    drop_cols = [c for c in ["SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV"] if c in df.columns]
    df_model = df.drop(columns=drop_cols, errors="ignore").copy()

    # Separate X, y
    y = df_model[target_col].astype(int)
    X = df_model.drop(columns=[target_col])

    # Identify column types
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number", "float", "int"]).columns.tolist()

    # Train/validation split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # ---- 2. Preprocessing ----
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )

    # ---- 3. Define models ----
    models_def = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=None,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=42,
        ),
    }

    if HAS_XGB:
        models_def["XGBoost"] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

    # ---- 4. Train & evaluate ----
    models = {}
    rows = []

    for name, clf in models_def.items():
        pipe = Pipeline(
            steps=[
                ("pre", preprocessor),
                ("clf", clf),
            ]
        )
        pipe.fit(X_train, y_train)

        # Probabilities & predictions for validation set
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, proba)
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)

        rows.append(
            {
                "Model": name,
                "ROC_AUC": roc_auc,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
            }
        )

        models[name] = pipe  # store the trained model

    metrics_df = pd.DataFrame(rows).set_index("Model").sort_values(
        by="ROC_AUC", ascending=False
    )

    best_name = metrics_df.index[0]
    best_model = models[best_name]

    return {
        "models": models,
        "summary": metrics_df,  # 👈 matches your tab3 code
        "best_name": best_name,
        "best_model": best_model,
        "X_test": X_test,
        "y_test": y_test,
    }
# --------------------------------------------------------------------
# MAIN APP
# --------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Analysis Dashboard",
    layout="wide",
)


st.title("Credit Risk Analysis Dashboard")

pipe_german, meta_german = load_german_model()
german_df = load_german_data()
home_df = load_home_credit_sample()
home_result = train_home_models(home_df) if home_df is not None else None

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "EDA (Home Credit)", "Model Metrics", "Single Prediction", "Batch Prediction"]
)

# --------------------------------------------------------------------
# TAB 1: OVERVIEW / DOCUMENTATION
# --------------------------------------------------------------------
with tab1:
    st.subheader("What This App Does")

    st.markdown(
        """
This Streamlit web application demonstrates how **data science and machine learning** can be 
used to **assess credit risk** and support lending decisions.

We work with **two real-world credit datasets**:

- 🟥 **German Credit (UCI)** – a compact dataset with 1,000 customers.  
  We trained and saved a classic **Logistic Regression** pipeline on this data (used in the midterm).  
  It still powers the basic *single* and *batch* prediction features.

- 🟦 **Home Credit Default Risk (Kaggle, sampled)** – a much larger, richer dataset describing 
  thousands of loan applications with financial, demographic, and behavioral features.
  We use a **sample** of the training data (stored as `home_credit_sample.csv`) to:
  - perform deeper **Exploratory Data Analysis (EDA)**
  - train three models:
      * Logistic Regression  
      * Random Forest  
      * XGBoost (if available in the environment)
  - compare model performance and visualize confusion matrices
  - make predictions for **individual clients** and **uploaded CSV batches**

---

### Problem Statement

For a lender, the central question is:

> **“Given an applicant’s profile, what is the probability that they will default on their loan?”**

Incorrectly approving a high-risk customer leads to losses; incorrectly rejecting a good customer 
means missed revenue. This app explores how different models and features can help balance those risks.

---

### How to Navigate the App

- **EDA (Home Credit)** – Explore distributions, target comparisons, and correlations interactively.  
- **Model Metrics** – Compare Logistic Regression, Random Forest, and XGBoost on the Home Credit sample.  
- **Single Prediction** – Enter features for one applicant and see estimated default probability.  
- **Batch Prediction** – Download a CSV template, fill it with multiple applicants, upload it, and
  receive a scored file with default probabilities.

Use this app as a **mini end-to-end project**: from data understanding → feature engineering → model
comparison → interactive deployment.
"""
    )

# --------------------------------------------------------------------
# TAB 2: EDA – HOME CREDIT
# --------------------------------------------------------------------
with tab2:
    st.subheader("Exploratory Data Analysis – Home Credit Sample")

    if home_df is None:
        st.warning("Home Credit sample could not be loaded. EDA is limited.")
    else:
        df = home_df.copy()

        target_col = "TARGET" if "TARGET" in df.columns else None
        if target_col:
            st.caption("Target column: `TARGET` (1 = default, 0 = no default).")
        else:
            st.warning(
                "No `TARGET` column in the sample – EDA will still work, but cannot be colored by default status."
            )

        # --- Basic info cards ---
        st.markdown("#### Basic Information")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Rows", df.shape[0])
        with c2:
            st.metric("Columns", df.shape[1])
        with c3:
            st.metric("Missing values", int(df.isna().sum().sum()))

        # --- Summary statistics with backwards-compatible describe ---
        with st.expander("Show summary statistics"):
            try:
                # Newer pandas
                st.write(df.describe(include="all", datetime_is_numeric=True).T)
            except TypeError:
                # Older pandas fallback
                st.write(df.describe(include="all").T)

        # --- Column summary table ---
        st.markdown("#### Column Overview")

        col_info = pd.DataFrame(
            {
                "dtype": df.dtypes,
                "missing_%": df.isna().mean() * 100,
                "n_unique": df.nunique(),
            }
        ).sort_values("missing_%", ascending=False)

        st.dataframe(col_info)

        # Identify numeric & categorical
        num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        st.markdown("### Interactive Plots")

        viz_type = st.radio(
            "Choose an analysis type:",
            [
                "Numeric distribution",
                "Categorical distribution",
                "Relationship with target",
                "Correlation heatmap",
                "Feature vs. feature (scatter)",
            ],
        )

        # --------------------------------------------------------------
        # Numeric distribution
        # --------------------------------------------------------------
        if viz_type == "Numeric distribution":
            if not num_cols:
                st.info("No numeric columns found.")
            else:
                col_num = st.selectbox("Choose numeric feature", num_cols)
                nbins = st.slider("Number of bins", 10, 80, 30)

                if target_col:
                    fig = px.histogram(
                        df,
                        x=col_num,
                        color=target_col,
                        nbins=nbins,
                        barmode="overlay",
                        marginal="box",
                        opacity=0.75,
                    )
                else:
                    fig = px.histogram(df, x=col_num, nbins=nbins, marginal="box")

                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)

        # --------------------------------------------------------------
        # Categorical distribution
        # --------------------------------------------------------------
        elif viz_type == "Categorical distribution":
            if not cat_cols:
                st.info("No categorical columns found.")
            else:
                col_cat = st.selectbox("Choose categorical feature", cat_cols)
                top_n = st.slider("Show top N categories", 5, 30, 15)

                temp = (
                    df[col_cat]
                    .value_counts(dropna=False)
                    .head(top_n)
                    .rename_axis(col_cat)
                    .reset_index(name="count")
                )
                temp["percent"] = temp["count"] / temp["count"].sum() * 100

                fig = px.bar(
                    temp,
                    x=col_cat,
                    y="count",
                    text="percent",
                    labels={"count": "Count"},
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_layout(xaxis_tickangle=-45, height=450)
                st.plotly_chart(fig, use_container_width=True)

        # --------------------------------------------------------------
        # Relationship with target
        # --------------------------------------------------------------
        elif viz_type == "Relationship with target":
            if not target_col:
                st.info("TARGET column not found; cannot compare classes.")
            else:
                # Only keep numeric columns with some variance
                candidate_cols = [c for c in num_cols if c != target_col]
                if not candidate_cols:
                    st.info("No suitable numeric columns found.")
                else:
                    col_num = st.selectbox("Numeric feature", candidate_cols)
                    plot_kind = st.radio(
                        "Plot type", ["Boxplot", "Violin", "KDE (2 curves)"], horizontal=True
                    )

                    if plot_kind in ["Boxplot", "Violin"]:
                        fig = px.box(
                            df,
                            x=target_col,
                            y=col_num,
                            points="all" if plot_kind == "Boxplot" else "outliers",
                        )
                        fig.update_layout(height=450)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # KDE curves – use seaborn + matplotlib
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.kdeplot(
                            data=df,
                            x=col_num,
                            hue=target_col,
                            common_norm=False,
                            ax=ax,
                        )
                        ax.set_title(f"Density of {col_num} by TARGET")
                        st.pyplot(fig)

        # --------------------------------------------------------------
        # Correlation heatmap (numeric)
        # --------------------------------------------------------------
        elif viz_type == "Correlation heatmap":
            if not num_cols:
                st.info("No numeric columns to correlate.")
            else:
                corr = df[num_cols].corr()

                if target_col in corr.columns:
                    # Show features most correlated with TARGET
                    k = st.slider(
                        "Show top K features (by absolute correlation with TARGET)",
                        5,
                        min(25, len(num_cols)),
                        min(10, len(num_cols)),
                    )
                    target_corr = corr[target_col].abs().sort_values(ascending=False)
                    top_features = target_corr.head(k).index.tolist()
                    corr_view = corr.loc[top_features, top_features]
                else:
                    corr_view = corr

                fig = px.imshow(
                    corr_view,
                    text_auto=".2f",
                    color_continuous_scale="RdBu",
                    zmin=-1,
                    zmax=1,
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

        # --------------------------------------------------------------
        # Feature vs feature scatter
        # --------------------------------------------------------------
        else:
            if len(num_cols) < 2:
                st.info("Need at least two numeric columns for scatter plot.")
            else:
                x_col = st.selectbox("X axis", num_cols, index=0)
                y_col = st.selectbox("Y axis", num_cols, index=1)
                sample_n = st.slider(
                    "Sample size (for speed)", 500, min(5000, len(df)), 2000, step=500
                )

                df_sample = df.sample(sample_n, random_state=42)

                if target_col:
                    fig = px.scatter(
                        df_sample,
                        x=x_col,
                        y=y_col,
                        color=target_col,
                        opacity=0.6,
                    )
                else:
                    fig = px.scatter(df_sample, x=x_col, y=y_col, opacity=0.6)

                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------
# TAB 3: MODEL METRICS – HOME CREDIT
# --------------------------------------------------------------------
with tab3:
    st.subheader("Model Performance – Home Credit Sample")

    if home_result is None:
        st.warning("Home Credit models are not available (missing dataset or training error).")
    else:
        metrics_df = home_result["summary"]

        st.markdown(
            """
These metrics are computed on a **hold-out validation set** from the Home Credit sample.  
Higher **ROC AUC** indicates a model that better separates default vs. non-default cases.
"""
        )

        st.dataframe(metrics_df.round(3))

        # Choose model for confusion matrix
        model_name = st.selectbox("Select a model to inspect", metrics_df.index.tolist())
        model = home_result["models"][model_name]
        X_test = home_result["X_test"]
        y_test = home_result["y_test"]

        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        st.markdown("#### Confusion Matrix")
        cm_df = pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])

        fig = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show classification report text"):
            report = classification_report(y_test, y_pred, digits=3)
            st.text(report)


# --------------------------------------------------------------------
# TAB 4: SINGLE PREDICTION – USE BEST HOME CREDIT MODEL
# --------------------------------------------------------------------
with tab4:
    st.subheader("Single Applicant Prediction (Home Credit Model)")

    if home_result is None or home_df is None:
        st.info(
            "Home Credit models are not available. Falling back to German Credit pipeline "
            "from the midterm."
        )

        # ---- Fallback: German Credit single prediction ----
        st.markdown("#### German Credit – Single Prediction (Fallback)")

        X_cols = meta_german["all_cols"]
        cat_cols = meta_german["cat_cols"]
        num_cols = meta_german["num_cols"]

        input_data = {}
        st.caption("Enter values for a single German Credit applicant:")

        for col in X_cols:
            if col in num_cols:
                val = st.number_input(f"{col}", value=0.0)
            else:
                val = st.text_input(f"{col}")
            input_data[col] = [val]

        if st.button("Predict (German Model)"):
            df_input = pd.DataFrame(input_data)
            prob = pipe_german.predict_proba(df_input)[:, 1][0]
            st.metric("Default Probability", f"{prob:.2%}")

    else:
        best_model = home_result["best_model"]
        feature_cols = home_result["feature_cols"]

        st.markdown(
            f"""
Using **best Home Credit model**: **{home_result['best_name']}**  
Enter feature values for **one applicant**. For categorical fields, use one of the values
that appear in the training data.
"""
        )

        df_ref = home_df[feature_cols]

        input_data = {}
        with st.form("single_home_form"):
            for col in feature_cols:
                series = df_ref[col]

                if pd.api.types.is_numeric_dtype(series):
                    default_val = float(series.median()) if series.notna().any() else 0.0
                    val = st.number_input(col, value=default_val)
                else:
                    choices = sorted(series.dropna().astype(str).unique().tolist())
                    if not choices:
                        val = ""
                    else:
                        val = st.selectbox(col, choices, index=0)
                input_data[col] = [val]

            submitted = st.form_submit_button("Predict default probability")

        if submitted:
            df_input = pd.DataFrame(input_data)
            prob = best_model.predict_proba(df_input)[:, 1][0]
            st.metric("Default Probability", f"{prob:.2%}")


# --------------------------------------------------------------------
# TAB 5: BATCH PREDICTION – HOME CREDIT BEST MODEL
# --------------------------------------------------------------------
with tab5:
    st.subheader("Batch Prediction – Upload CSV of Applicants")

    if home_result is None or home_df is None:
        st.info(
            "Home Credit models are not available. You can still run batch prediction "
            "using the separate German Credit app or notebook."
        )
    else:
        best_model = home_result["best_model"]
        feature_cols = home_result["feature_cols"]

        st.markdown(
            """
1. **Download a CSV template** with the correct columns.  
2. Fill in rows for new applicants (keep column names unchanged).  
3. Upload the completed file to get predicted default probabilities.
"""
        )

        # --- Template download ---
        template_df = home_df[feature_cols].head(200).copy()
        template_bytes = template_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download batch template (CSV)",
            data=template_bytes,
            file_name="home_credit_batch_template.csv",
            mime="text/csv",
        )

        # --- File uploader ---
        uploaded_file = st.file_uploader(
            "Upload CSV with the same columns as the template", type=["csv"]
        )

        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)

            missing = set(feature_cols) - set(batch_df.columns)
            extra = set(batch_df.columns) - set(feature_cols)

            if missing:
                st.error(f"The uploaded file is missing columns: {missing}")
            else:
                if extra:
                    st.warning(
                        f"Extra columns in file will be ignored: {extra}"
                    )
                    batch_df = batch_df[feature_cols]

                probs = best_model.predict_proba(batch_df)[:, 1]
                result_df = batch_df.copy()
                result_df["default_probability"] = probs

                st.markdown("#### Preview of scored file")
                st.dataframe(result_df.head())

                out_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results as CSV",
                    data=out_bytes,
                    file_name="home_credit_batch_scored.csv",
                    mime="text/csv",
                )
