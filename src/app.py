import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import plotly.express as px

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
HERE = Path(__file__).resolve().parent       # .../src
ROOT = HERE.parent                           # .../cmse830_fds
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

DATA_PATH_CREDIT = DATA_DIR / "credit-g.csv"
DATA_PATH_HOME = DATA_DIR / "home_credit_sample.csv"

# Prefer the new Home Credit sample; fall back to old German credit if needed
if DATA_PATH_HOME.exists():
    ACTIVE_DATA_PATH = DATA_PATH_HOME
    ACTIVE_DATA_NAME = "Home Credit Sample (Kaggle)"
else:
    ACTIVE_DATA_PATH = DATA_PATH_CREDIT
    ACTIVE_DATA_NAME = "German Credit (UCI)"

MODEL_PATH = MODELS_DIR / "credit_pipeline.joblib"
META_PATH = MODELS_DIR / "metadata.joblib"


def require_file(p: Path, label: str):
    """
    Small helper: makes sure a file exists and gives a nice error if not.
    """
    if not p.exists():
        raise FileNotFoundError(f"{label} not found at: {p}")
    return p


# ---------------------------------------------------
# Load model + metadata
# ---------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    pipe = joblib.load(require_file(MODEL_PATH, "Model"))
    meta = joblib.load(require_file(META_PATH, "Metadata"))
    return pipe, meta


# ---------------------------------------------------
# Load dataset used for EDA
# ---------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(ACTIVE_DATA_PATH)
    return df


pipe, meta = load_model()
df = load_data()

# Try to guess the target column (for metrics & some plots)
if "TARGET" in df.columns:
    TARGET_COL = "TARGET"
elif "class" in df.columns:
    TARGET_COL = "class"
else:
    TARGET_COL = None

st.title("Credit Risk Analysis Dashboard")

# ---------------------------------------------------
# Tabs
# ---------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "EDA", "Model Metrics", "Single Prediction", "Batch Prediction"]
)

# ---------------------------------------------------
# Tab 1: Overview
# ---------------------------------------------------
with tab1:
    st.subheader("What This App Does")

    st.markdown(
        f"""
**Dataset in use:** `{ACTIVE_DATA_NAME}`  

This dashboard is a mini **credit risk / default prediction system**.  
It demonstrates a realistic workflow that a bank or lending company might use:

- Load a real-world credit dataset (now using a **larger Home Credit sample**).
- Perform **initial data analysis and EDA** to understand applicants and risk drivers.
- Train a **machine learning model** to predict probability of default.
- Make **single-applicant predictions** and **batch predictions** from uploaded CSVs.
- View **model performance metrics** for transparency.

During the project I experimented with multiple models:

- **Logistic Regression** (baseline, interpretable model)
- **Tree-based models** like **Random Forest** and **Gradient Boosting / XGBoost** (more flexible, non-linear)
- Compared them using metrics like **ROC AUC, accuracy, precision, recall, and F1**.

For deployment, the app currently uses the trained pipeline saved as  
`models/credit_pipeline.joblib` with preprocessing + classifier bundled together.
        """
    )

    st.info(
        "Note: The app is designed so the backend model can be swapped out later "
        "without changing the Streamlit UI. Only the saved pipeline and metadata need updating."
    )

# ---------------------------------------------------
# Tab 2: EDA
# ---------------------------------------------------
with tab2:
    st.subheader("Exploratory Data Analysis (EDA)")

    st.markdown(
        f"""
We are currently exploring: **{ACTIVE_DATA_NAME}**  

Below are a few quick views to understand the structure of the data and check for patterns
that might be related to credit risk.
        """
    )

    st.write("**Dataset preview**")
    st.dataframe(df.head())

    # Detect numeric and categorical columns for EDA
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    # 1) Target distribution
    if TARGET_COL is not None and TARGET_COL in df.columns:
        st.write(f"**Target distribution: `{TARGET_COL}`**")
        fig, ax = plt.subplots()
        df[TARGET_COL].value_counts().plot(kind="bar", ax=ax)
        ax.set_xlabel(TARGET_COL)
        ax.set_ylabel("Count")
        st.pyplot(fig)
    else:
        st.warning("Target column not found – cannot show target distribution.")

    # 2) Numeric column histogram (selectable)
    if num_cols:
        st.write("**Histogram of a numeric feature**")
        col_choice = st.selectbox("Choose a numeric column", num_cols)
        fig, ax = plt.subplots()
        ax.hist(df[col_choice].dropna(), bins=40)
        ax.set_xlabel(col_choice)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    # 3) Correlation heatmap (numeric variables)
    if len(num_cols) >= 2:
        st.write("**Correlation heatmap (numeric features)**")
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # 4) Simple Plotly scatter (numeric vs target / another numeric)
    if len(num_cols) >= 2:
        st.write("**Interactive scatter plot (Plotly)**")
        x_var = st.selectbox("X axis", num_cols, index=0, key="eda_x")
        y_var = st.selectbox("Y axis", num_cols, index=1, key="eda_y")

        color_opt = TARGET_COL if TARGET_COL in df.columns else None
        fig_scatter = px.scatter(
            df.sample(min(len(df), 5000)),  # sample to keep it light
            x=x_var,
            y=y_var,
            color=color_opt,
            title=f"{x_var} vs {y_var}",
            opacity=0.6,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------------------------------
# Tab 3: Model Metrics
# ---------------------------------------------------
with tab3:
    st.subheader("Model Performance Metrics")

    if TARGET_COL is None or TARGET_COL not in df.columns:
        st.warning(
            "Could not detect a target column (e.g., 'TARGET' or 'class'). "
            "Metrics cannot be computed."
        )
    else:
        st.markdown(
            """
These metrics are computed by evaluating the **saved pipeline** on the current dataset.
This is mainly to get a sense of how well the model separates good vs. bad credit.

*(In a more rigorous setup, we would compute these on a separate test or validation set.)*
            """
        )

        # Build X, y according to dataset
        if TARGET_COL == "class":
            # German dataset: 'good' / 'bad'
            y = (df[TARGET_COL] == "bad").astype(int)
        else:
            # Home Credit: 0/1
            y = df[TARGET_COL].astype(int)

        X = df.drop(columns=[TARGET_COL])

        # Predict probabilities and labels
        try:
            y_proba = pipe.predict_proba(X)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            metrics = {
                "ROC AUC": roc_auc_score(y, y_proba),
                "Accuracy": accuracy_score(y, y_pred),
                "Precision": precision_score(y, y_pred, zero_division=0),
                "Recall": recall_score(y, y_pred, zero_division=0),
                "F1-score": f1_score(y, y_pred, zero_division=0),
            }

            st.write("**Summary metrics**")
            st.dataframe(
                pd.DataFrame(metrics, index=["Model"]).T.rename(columns={"Model": "Value"})
            )

            # Confusion matrix
            cm = confusion_matrix(y, y_pred)
            st.write("**Confusion matrix**")
            cm_df = pd.DataFrame(
                cm,
                index=["Actual 0 (no default)", "Actual 1 (default)"],
                columns=["Pred 0", "Pred 1"],
            )
            st.dataframe(cm_df)

        except Exception as e:
            st.error(
                f"Could not compute metrics from the pipeline. "
                f"Error: {e}"
            )

# ---------------------------------------------------
# Tab 4: Single Prediction
# ---------------------------------------------------
with tab4:
    st.subheader("Single Applicant Prediction")

    st.markdown(
        """
Fill in the applicant’s information below.  
The app will run the input through the same preprocessing + model pipeline
and output a **default probability**.
        """
    )

    input_data = {}

    # Use metadata to decide which columns are numeric/categorical
    all_cols = meta.get("all_cols", [])
    cat_cols = meta.get("cat_cols", [])
    num_cols_meta = meta.get("num_cols", [])

    # Use df to get sensible defaults / choices where possible
    for col in all_cols:
        if col in num_cols_meta:
            default_val = (
                float(df[col].median()) if col in df.columns and df[col].notna().any() else 0.0
            )
            val = st.number_input(col, value=default_val)
            input_data[col] = [val]
        else:
            # categorical – use unique values from df if available
            if col in df.columns:
                options = sorted(df[col].dropna().unique().tolist())
                if len(options) > 0:
                    val = st.selectbox(col, options)
                else:
                    val = st.text_input(col)
            else:
                val = st.text_input(col)
            input_data[col] = [val]

    if st.button("Predict default risk"):
        df_input = pd.DataFrame(input_data)
        try:
            prob = pipe.predict_proba(df_input)[:, 1][0]
            st.metric("Predicted default probability", f"{prob:.2%}")
            if prob > 0.5:
                st.warning("Model view: **High risk** of default.")
            else:
                st.success("Model view: **Lower risk** of default.")
        except Exception as e:
            st.error(f"Could not generate prediction. Error: {e}")

# ---------------------------------------------------
# Tab 5: Batch Prediction
# ---------------------------------------------------
with tab5:
    st.subheader("Batch CSV Prediction")

    st.markdown(
        """
Upload a CSV with the same feature columns that the model expects  
(the same schema as the training data, without the target column).  

The app will return a file with an extra column containing **default probability**.
        """
    )

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            st.write("Input preview:")
            st.dataframe(batch_df.head())

            probs = pipe.predict_proba(batch_df)[:, 1]
            batch_df["default_probability"] = probs

            st.write("Preview with predictions:")
            st.dataframe(batch_df.head())

            # Download link
            csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download predictions as CSV",
                data=csv_bytes,
                file_name="batch_predictions.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Error while scoring batch file: {e}")
