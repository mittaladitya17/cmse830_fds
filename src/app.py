from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import plotly.express as px

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

GERMAN_PATH = DATA_DIR / "credit-g.csv"          # model dataset
HOME_PATH   = DATA_DIR / "home_credit_sample.csv"  # bigger sample

MODEL_PATH = MODELS_DIR / "credit_pipeline.joblib"
META_PATH  = MODELS_DIR / "metadata.joblib"


def require_file(p: Path, label: str):
    if not p.exists():
        raise FileNotFoundError(f"{label} not found at: {p}")
    return p


@st.cache_resource(show_spinner=False)
def load_data():
    df_german = pd.read_csv(require_file(GERMAN_PATH, "German credit data"))
    df_home   = pd.read_csv(require_file(HOME_PATH, "Home Credit sample"))
    return df_german, df_home


@st.cache_resource(show_spinner=False)
def load_model():
    pipe = joblib.load(require_file(MODEL_PATH, "Model"))
    meta = joblib.load(require_file(META_PATH, "Metadata"))
    return pipe, meta


df_german, df_home = load_data()
pipe, meta = load_model()


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
    st.subheader("Exploratory Data Analysis (Home Credit Sample)")

    st.write("Shape:", df_home.shape)
    st.write("Preview:")
    st.dataframe(df_home.head())

    st.markdown("### Target distribution")
    if "TARGET" in df_home.columns:
        fig, ax = plt.subplots()
        df_home["TARGET"].value_counts().plot(kind="bar", ax=ax)
        ax.set_xlabel("Default (1 = yes, 0 = no)")
        st.pyplot(fig)

    st.markdown("### Numeric correlations")
    num_cols = df_home.select_dtypes(include="number").columns
    corr = df_home[num_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, ax=ax)
    st.pyplot(fig)

    st.markdown("### Example Plotly scatter")
    if "AMT_CREDIT" in df_home.columns and "AMT_INCOME_TOTAL" in df_home.columns:
        fig = px.scatter(
            df_home.sample(min(3000, len(df_home))), 
            x="AMT_INCOME_TOTAL", 
            y="AMT_CREDIT",
            color="TARGET" if "TARGET" in df_home.columns else None,
            title="Credit Amount vs Income"
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------
# Tab 3: Model Metrics
# ---------------------------------------------------
with tab3:
    st.subheader("Model Performance Metrics")

    st.markdown(
        """
        These metrics are computed using the **German Credit** dataset, which is the data
        the current logistic regression model was trained on.

        In a more rigorous setup, we would compute these on a separate validation or test set.
        """
    )

    try:
        # y: 1 = bad credit, 0 = good
        y = (df_german["class"] == "bad").astype(int)

        # X: make sure we pass exactly the columns the pipeline expects
        X = df_german[meta["all_cols"]]

        proba = pipe.predict_proba(X)[:, 1]
        y_pred = (proba >= 0.5).astype(int)

        roc = roc_auc_score(y, proba)
        acc = (y_pred == y).mean()

        c1, c2 = st.columns(2)
        with c1:
            st.metric("ROC AUC", f"{roc:.3f}")
        with c2:
            st.metric("Accuracy", f"{acc:.3f}")

        st.markdown("### Classification Report")
        st.text(classification_report(y, y_pred, digits=3))

        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Could not compute metrics from the pipeline. Error: {e}")

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
