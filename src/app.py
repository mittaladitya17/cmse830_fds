# --- app.py ---
import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os, joblib
import os
from pathlib import Path
import plotly.express as px


HERE = Path(__file__).resolve().parent      # .../finance-risk-dashboard/src
ROOT = HERE.parent                          # .../finance-risk-dashboard
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

DATA_PATH = DATA_DIR / "home_credit_sample.csv"   # new bigger dataset
MODEL_PATH = MODELS_DIR / "credit_pipeline.joblib"
META_PATH  = MODELS_DIR / "metadata.joblib"

def require_file(p: Path, label: str):
    if not p.exists():
        raise FileNotFoundError(f"{label} not found at: {p}")
    return p

st.title("Credit Risk Analysis Dashboard")
st.markdown("""
This dashboard predicts **loan default risk** using the German Credit dataset.
It includes data exploration, visualization, and a simple ML model built with Logistic Regression.
""")

@st.cache_resource(show_spinner=False)
def load_model():
    pipe = joblib.load(require_file(MODEL_PATH, "Model"))
    meta = joblib.load(require_file(META_PATH, "Metadata"))
    return pipe, meta


pipe, meta = load_model()

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "EDA", "Single Prediction", "Batch Prediction"])

# --- Tab 1: Project Overview ---
with tab1:
    st.subheader("What This App Does")
    st.markdown("""
    - Predict whether a loan applicant is likely to default.
    - Explore key features and patterns in the dataset through interactive EDA.
    - Test the ML model on single applicants or batch files.
    - Understand model performance metrics and feature importance (coming soon!).
    """)

    st.subheader("Dataset Info")
    st.markdown("""
    - Source: [UCI / OpenML German Credit Dataset](https://www.openml.org/d/31)  
    - Records: ~1000 applicants  
    - Target: `good` (repays) or `bad` (default)  
    - Features: Demographics, financial status, loan attributes, etc.
    """)

    with st.expander("Modeling Notes"):
        st.markdown("""
        - Model: **Logistic Regression** (`class_weight='balanced'`, `max_iter=1000`)  
        - Preprocessing: OneHotEncoder (categoricals), StandardScaler (numerics)  
        - Train/test split with stratification; primary metric: **ROC AUC**  
        - Threshold tuning and feature importance coming soon.
        """)

    with st.expander("Roadmap (Next 50%)"):
        st.markdown("""
        - Feature importance (coefficients, SHAP)  
        - Model comparison (RandomForest, XGBoost)  
        - Cost-sensitive thresholds for business KPIs  
        - Deploy to Hugging Face / Streamlit Cloud  
        """)


# ---- Tab 2: EDA + Model Metrics ----
with tab2:
    st.subheader("Exploratory Data Analysis (Home Credit Sample)")

    # load data
    df = pd.read_csv(DATA_PATH)

    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.dataframe(df.head())

    # target distribution
    if "TARGET" in df.columns:
        st.markdown("### Target Distribution (0 = repaid, 1 = default)")
        fig = px.histogram(df, x="TARGET")
        st.plotly_chart(fig, use_container_width=True)

    # numeric correlation heatmap (first 20 numeric cols to keep it light)
    st.markdown("### Correlation Heatmap (numeric features)")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    num_subset = num_cols[:20]   # don't plot all, just a subset
    corr = df[num_subset].corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto")
    st.plotly_chart(fig_corr, use_container_width=True)

    # some feature vs TARGET plots (only if columns exist)
    st.markdown("### Key Features vs Default Probability")

    if {"AMT_CREDIT", "TARGET"} <= set(df.columns):
        fig_credit = px.histogram(
            df, x="AMT_CREDIT", color="TARGET", barmode="overlay",
            nbins=50,
            title="Credit Amount vs Target"
        )
        fig_credit.update_layout(xaxis_title="AMT_CREDIT", yaxis_title="Count")
        st.plotly_chart(fig_credit, use_container_width=True)

    if {"AMT_INCOME_TOTAL", "TARGET"} <= set(df.columns):
        fig_income = px.histogram(
            df, x="AMT_INCOME_TOTAL", color="TARGET", barmode="overlay",
            nbins=50,
            title="Income vs Target"
        )
        fig_income.update_layout(xaxis_title="AMT_INCOME_TOTAL", yaxis_title="Count")
        st.plotly_chart(fig_income, use_container_width=True)

    # ----- Model metrics from training -----
    st.markdown("### Model Performance (Saved from Training Script)")

    metrics = meta.get("metrics", {})
    if metrics:
        metrics_df = pd.DataFrame(metrics).T  # models as rows
        st.dataframe(metrics_df)

        # simple bar chart of ROC AUC
        if "roc_auc" in metrics_df.columns:
            fig_auc = px.bar(
                metrics_df.reset_index(),
                x="index",
                y="roc_auc",
                title="ROC AUC by Model"
            )
            fig_auc.update_layout(xaxis_title="Model", yaxis_title="ROC AUC")
            st.plotly_chart(fig_auc, use_container_width=True)

        st.info(f"Best model (by ROC AUC): **{meta.get('best_model', 'unknown')}**")
    else:
        st.warning("No metric information found in metadata.")


# --- Tab 3 ---
with tab3:
    st.subheader("Single Applicant Prediction")
    input_data = {}
    for col in meta["all_cols"]:
        input_data[col] = [st.text_input(f"Enter {col}:")]

    if st.button("Predict"):
        df_input = pd.DataFrame(input_data)
        prob = pipe.predict_proba(df_input)[:, 1][0]
        st.metric("Default Probability", f"{prob:.2%}")

# --- Tab 4: Batch Prediction ---
with tab4:
    st.header("Batch Prediction")

    # load full dataset once for examples / schema
    df_full = pd.read_csv(require_file(DATA_PATH, "Dataset"))
    X_full = df_full.drop(columns=["class"])

    st.markdown("**Required columns (exact names):**")
    st.code(", ".join(meta["all_cols"]), language="text")

    # ---- Download a VALID sample (5 real rows, correct types) ----
    st.subheader("Download Sample CSV (valid schema)")
    sample_df = X_full.head(5).copy()  # 5 valid rows (no target column)
    st.download_button(
        "Download sample_batch.csv",
        data=sample_df.to_csv(index=False),
        file_name="sample_batch.csv",
        mime="text/csv",
        help="Use this as a starting point"
    )

    st.write("---")
    st.subheader("Upload Your CSV")

    file = st.file_uploader("Upload CSV with the columns above", type=["csv"])

    def coerce_schema(df_in: pd.DataFrame) -> pd.DataFrame:
        """Ensure uploaded data matches training schema: columns, order, dtypes."""
        df = df_in.copy()

        # Add any missing columns as NaN; ignore extra columns
        for col in meta["all_cols"]:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[meta["all_cols"]]  # reorder / drop extras

        # Split by type
        cat_cols = meta["cat_cols"]
        num_cols = meta["num_cols"]

        # Coerce numerics
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Fill missing: simple defaults (your pipeline can still handle NaN, but this reduces errors)
        df[cat_cols] = df[cat_cols].fillna("unknown")
        for col in num_cols:
            # use training medians as safe numeric imputation (fallback to 0 if all NaN)
            median_val = pd.to_numeric(X_full[col], errors="coerce").median()
            df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)

        # Ensure categorical are strings
        for col in cat_cols:
            df[col] = df[col].astype(str)

        return df

    if file:
        try:
            user_df_raw = pd.read_csv(file)
            st.write("**Uploaded preview:**")
            st.dataframe(user_df_raw.head())

            # Validate at least required columns exist
            missing = [c for c in meta["all_cols"] if c not in user_df_raw.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
                st.stop()

            # Coerce to training schema + dtypes
            user_df = coerce_schema(user_df_raw)

            # Predict probabilities
            probs = pipe.predict_proba(user_df)[:, 1]
            results = user_df.copy()
            results["default_risk_prob"] = probs

            st.success("✅ Scoring complete!")
            st.dataframe(results.head())

            st.download_button(
                "Download results.csv",
                data=results.to_csv(index=False),
                file_name="results.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Could not score file: {e}")
            st.info("Tip: Start from the **Download sample_batch.csv** above and edit values.")


