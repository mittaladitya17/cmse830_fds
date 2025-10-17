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

DATA_PATH  = DATA_DIR / "credit-g.csv"
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


# --- Tab 2: EDA ---
with tab2:
    st.subheader("Dataset Preview")

    # NOTE: these two lines must be indented exactly 4 spaces under 'with tab2:'
    df = pd.read_csv(require_file(DATA_PATH, "Dataset"))
    st.dataframe(df.head())

    st.subheader("Class Distribution (Interactive)")
    class_counts = df["class"].value_counts().reset_index()
    class_counts.columns = ["class", "count"]
    fig = px.bar(class_counts, x="class", y="count", text="count", title="Good vs Bad Credit")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Numeric Feature Explorer")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    num_col = st.selectbox("Pick a numeric column", num_cols, index=0)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x=num_col, color="class", barmode="overlay",
                       marginal="rug", nbins=40, title=f"Histogram: {num_col}")
    st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(df, x="class", y=num_col, points="all", title=f"Boxplot by class: {num_col}")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Class Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="class", data=df, ax=ax, palette="viridis")
    ax.set_title("Good vs Bad Credit")
    st.pyplot(fig)

    st.subheader("Correlation Heatmap (numeric)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include="number").corr(),
                annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlations")
    st.pyplot(fig)

    st.subheader("Correlation Explorer (with target)")
    method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"], index=0)
    corr = df.select_dtypes(include="number").corr(method=method)
    if "class" in df.columns:
        st.info("Note: target is categorical; using a numeric proxy for quick ranking.")
# quick numeric proxy: map class to 0/1 for ranking only
    y_num = (df["class"] == "bad").astype(int)
    corr_with_target = df.select_dtypes(include="number").assign(_y=y_num).corr(method=method)["_y"].drop("_y")
    topN = st.slider("Show top N features by |correlation| with target", 3, min(15, len(corr_with_target)), 8)
    series = corr_with_target.abs().sort_values(ascending=False).head(topN)
    fig = px.bar(series, x=series.index, y=series.values,
             labels={"x":"feature", "y":f"|corr| with target ({method})"},
             title=f"Top {topN} Correlated Features")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Categorical Breakdown by Class")
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    if cat_cols:
        cat = st.selectbox("Pick a categorical column", cat_cols, index=0)
    counts = df.groupby([cat, "class"]).size().reset_index(name="count")
    fig = px.bar(counts, x=cat, y="count", color="class", barmode="stack",
                 title=f"{cat}: distribution by class")
    st.plotly_chart(fig, use_container_width=True)


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


