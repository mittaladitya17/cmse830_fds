import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


# ----------------------------
# Paths & helpers
# ----------------------------

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

CREDIT_PATH = DATA_DIR / "credit-g.csv"
HOME_SAMPLE_PATH = DATA_DIR / "home_credit_sample.csv"

MODEL_PATH = MODELS_DIR / "credit_pipeline.joblib"
META_PATH = MODELS_DIR / "metadata.joblib"


def require_file(p: Path, label: str) -> Path:
    """Small helper to fail loudly if a required file is missing."""
    if not p.exists():
        raise FileNotFoundError(f"{label} not found at: {p}")
    return p


@st.cache_resource(show_spinner=False)
def load_saved_pipeline():
    """Load the German Credit pipeline and metadata (trained offline)."""
    pipe = joblib.load(require_file(MODEL_PATH, "Model"))
    meta = joblib.load(require_file(META_PATH, "Metadata"))
    return pipe, meta


@st.cache_resource(show_spinner=False)
def load_datasets():
    """Load both datasets used in the app."""
    credit_df = pd.read_csv(require_file(CREDIT_PATH, "German Credit data"))
    # home_credit_sample.csv is a smaller slice of the big Kaggle dataset
    try:
        home_df = pd.read_csv(require_file(HOME_SAMPLE_PATH, "Home Credit sample"))
    except FileNotFoundError:
        home_df = None
    return credit_df, home_df


# ----------------------------
# Home Credit models (LR / RF / XGB)
# ----------------------------

@st.cache_resource(show_spinner=True)
def train_home_models(home_df: pd.DataFrame):
    """
    Train three models (LR, Random Forest, XGBoost) on the Home Credit sample.
    Returns a dict with metrics + the fitted models.
    """
    if home_df is None:
        return None

    df = home_df.copy()

    if "TARGET" not in df.columns:
        raise ValueError("Home Credit sample must contain a 'TARGET' column.")

    y = df["TARGET"]
    X = df.drop(columns=["TARGET"])

    # Identify feature types
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    # Basic preprocessing: impute + scale numeric, impute + one-hot encode categorical
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )

    # Models
    base_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            n_jobs=None
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=1,
            tree_method="hist"
        ),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    results = {}

    for name, clf in base_models.items():
        model_pipe = Pipeline(
            steps=[
                ("pre", preprocessor),
                ("clf", clf),
            ]
        )
        model_pipe.fit(X_train, y_train)

        prob = model_pipe.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)

        acc = accuracy_score(y_test, pred)
        auc = roc_auc_score(y_test, prob)

        results[name] = {
            "model": model_pipe,
            "accuracy": acc,
            "roc_auc": auc,
        }

    return results


# ----------------------------
# Layout
# ----------------------------

st.set_page_config(
    page_title="Credit Risk Dashboard",
    layout="wide"
)

st.title("Credit Risk Analysis Dashboard")

pipe, meta = load_saved_pipeline()
credit_df, home_df = load_datasets()
home_models = None
if home_df is not None:
    home_models = train_home_models(home_df)

tab_overview, tab_eda, tab_metrics, tab_single, tab_batch = st.tabs(
    ["Overview", "EDA", "Model Metrics", "Single Prediction", "Batch Prediction"]
)

# ----------------------------
# Tab 1: Overview
# ----------------------------

with tab_overview:
    st.subheader("What This App Does")

    st.markdown(
        """
This project is a **credit risk scoring and analysis dashboard**.

It has two main parts:

1. **German Credit Dataset (Smaller, Classic Dataset)**
   - Pre-trained **Logistic Regression pipeline** (trained offline and loaded from disk).
   - Used for the **Single Applicant** and **Batch Prediction** tools.
   - Shows how a production-style credit scoring model can be packaged and deployed.

2. **Home Credit Sample Dataset (Larger, Real-World Style Data)**
   - A sample from the much larger Kaggle *Home Credit Default Risk* dataset.
   - We train three models live inside the app:
     - Logistic Regression  
     - Random Forest  
     - XGBoost  
   - These models highlight **advanced modeling techniques** and how model choice impacts metrics.

The dashboard supports:

- **Interactive EDA** on both datasets  
- **Model performance comparison** (accuracy & ROC AUC)  
- **Single-loan risk scoring** for experimentation  
- **Batch CSV scoring & download** for a more realistic workflow
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

# ----------------------------
# Tab 2: Interactive EDA
# ----------------------------

with tab_eda:
    st.subheader("Exploratory Data Analysis (EDA)")

    dataset_choice = st.radio(
        "Choose dataset to explore:",
        ["German Credit", "Home Credit Sample"],
        horizontal=True
    )

    if dataset_choice == "German Credit":
        df = credit_df.copy()
        target_col = "class"
        st.caption("Target column: `class` (good / bad).")
    else:
        if home_df is None:
            st.error(
                "Home Credit sample not loaded. Please add `home_credit_sample.csv` "
                "to the `data/` folder."
            )
            df = None
            target_col = None
        else:
            df = home_df.copy()
            target_col = "TARGET" if "TARGET" in df.columns else None
            if target_col:
                st.caption("Target column: `TARGET` (1 = default, 0 = no default).")
            else:
                st.warning(
                    "No `TARGET` column found in Home Credit sample. "
                    "EDA will still work, but plots won't be colored by target."
                )

    if df is not None:
        st.markdown("#### Basic Info")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Rows", df.shape[0])
        with c2:
            st.metric("Columns", df.shape[1])
        with c3:
            st.metric("Missing values", int(df.isna().sum().sum()))

        with st.expander("Show summary statistics"):
            desc = df.describe(include="all")   
            st.write(desc.T)

        # Identify numeric / categorical columns
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        st.markdown("### 1. Univariate Distribution (Numeric)")

        if num_cols:
            col_num = st.selectbox(
                "Choose a numeric feature",
                num_cols,
                key=f"{dataset_choice}_num"
            )
            bins = st.slider(
                "Number of bins",
                min_value=10,
                max_value=60,
                value=30,
                step=5,
                key=f"{dataset_choice}_bins"
            )

            color_opt = target_col if (target_col in df.columns) else None

            fig_hist = px.histogram(
                df,
                x=col_num,
                nbins=bins,
                color=color_opt,
                marginal="box",
                opacity=0.8
            )
            fig_hist.update_layout(height=400, bargap=0.05)
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No numeric columns found for this dataset.")

        st.markdown("### 2. Categorical Counts")

        if cat_cols:
            col_cat = st.selectbox(
                "Choose a categorical feature",
                cat_cols,
                key=f"{dataset_choice}_cat"
            )

            color_opt = target_col if (target_col in df.columns) else None

            fig_cat = px.histogram(
                df,
                x=col_cat,
                color=color_opt,
                barmode="group"
            )
            fig_cat.update_xaxes(categoryorder="total descending")
            fig_cat.update_layout(height=400)
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("No categorical columns found for this dataset.")

        st.markdown("### 3. Bivariate Relationship (Scatter)")

        if len(num_cols) >= 2:
            col_x = st.selectbox(
                "X-axis",
                num_cols,
                key=f"{dataset_choice}_x"
            )
            col_y = st.selectbox(
                "Y-axis",
                num_cols,
                key=f"{dataset_choice}_y"
            )

            color_opt = target_col if (target_col in df.columns) else None

            fig_scatter = px.scatter(
                df,
                x=col_x,
                y=col_y,
                color=color_opt,
                opacity=0.7
            )
            fig_scatter.update_layout(height=420)
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Need at least two numeric columns for scatter plot.")

        st.markdown("### 4. Correlation Heatmap")

        if num_cols and st.checkbox("Show correlation heatmap", key=f"{dataset_choice}_corr"):
            corr = df[num_cols].corr()
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu",
                origin="lower"
            )
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)

# ----------------------------
# Tab 3: Model Metrics
# ----------------------------

with tab_metrics:
    st.subheader("Model Performance Metrics")

    st.markdown(
        """
These metrics are meant to **compare models and communicate performance**, not to be
perfectly optimized.  

- The **German Credit** model was trained offline and loaded from disk.  
- The **Home Credit** models (Logistic / Random Forest / XGBoost) are trained on the
  sample data when the app starts (and cached for speed).
        """
    )

    # --- German Credit metrics (using saved pipeline) ---
    st.markdown("### German Credit – Saved Logistic Regression Pipeline")

    try:
        X_credit = credit_df.drop(columns=["class"])
        y_credit = (credit_df["class"] == "bad").astype(int)

        prob_credit = pipe.predict_proba(X_credit)[:, 1]
        pred_credit = (prob_credit >= 0.5).astype(int)

        acc_credit = accuracy_score(y_credit, pred_credit)
        auc_credit = roc_auc_score(y_credit, prob_credit)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Accuracy", f"{acc_credit:.3f}")
        with c2:
            st.metric("ROC AUC", f"{auc_credit:.3f}")

    except Exception as e:
        st.error(f"Could not compute metrics for German Credit model: {e}")

    st.markdown("---")
    st.markdown("### Home Credit Sample – LR / RF / XGBoost")

    if home_models is None:
        st.info(
            "Home Credit models were not trained because the sample dataset "
            "could not be loaded."
        )
    else:
        rows = []
        for name, info in home_models.items():
            rows.append(
                {
                    "Model": name,
                    "Accuracy": round(info["accuracy"], 3),
                    "ROC AUC": round(info["roc_auc"], 3),
                }
            )

        metrics_df = pd.DataFrame(rows)
        st.dataframe(metrics_df, use_container_width=True)

        fig_bar = px.bar(
            metrics_df,
            x="Model",
            y=["Accuracy", "ROC AUC"],
            barmode="group"
        )
        fig_bar.update_layout(height=450)
        st.plotly_chart(fig_bar, use_container_width=True)

# ----------------------------
# Tab 4: Single Applicant Prediction
# ----------------------------

with tab_single:
    st.subheader("Single Applicant Prediction (German Credit)")

    st.markdown(
        """
Use this tab to **simulate a single loan application**.  
        """
    )

    input_data = {}
    for col in meta["all_cols"]:
        input_data[col] = [st.text_input(f"{col}", key=f"single_{col}")]

    if st.button("Predict Default Risk"):
        df_input = pd.DataFrame(input_data)

        try:
            prob = pipe.predict_proba(df_input)[:, 1][0]
            st.metric("Estimated Default Probability", f"{prob:.2%}")
            st.caption("Higher probability ⇒ higher risk of default.")
        except Exception as e:
            st.error(f"Could not make prediction. Error: {e}")

# ----------------------------
# Tab 5: Batch Prediction
# ----------------------------

with tab_batch:
    st.subheader("Batch Prediction on CSV")

    if home_df is None or home_models is None:
        st.info("Home Credit sample data was not loaded, so batch prediction is disabled.")
    else:
        # 1. Choose which trained model to use for batch prediction
        home_models = home_result["models"]
        best_name = home_result.get("best_name", "Logistic Regression")
        home_best_model = home_models[best_name]

        st.markdown(f"Using **{best_name}** model for batch prediction.")

        # 2. Offer a sample CSV for users to download
        st.markdown("#### Download Sample CSV")

        # Use a small sample of the Home Credit data, dropping the TARGET column
        sample_cols = home_result["feature_info"]["all_features"]
        sample_df = home_df[sample_cols].head(200)  # first 200 rows
        sample_csv = sample_df.to_csv(index=False)

        st.download_button(
            label="Download sample_batch.csv",
            data=sample_csv,
            file_name="sample_batch.csv",
            mime="text/csv",
            help="Download a sample file you can edit and upload below."
        )

        st.markdown("---")
        st.markdown("#### Upload Batch File")

        uploaded_file = st.file_uploader(
            "Upload a CSV with the same columns as the training data (no TARGET needed).",
            type=["csv"],
            key="batch_uploader"
        )

        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(batch_df.head())

                # 3. Align columns with what the model expects
                expected_cols = home_result["feature_info"]["all_features"]

                missing_in_upload = [c for c in expected_cols if c not in batch_df.columns]
                extra_in_upload = [c for c in batch_df.columns if c not in expected_cols]

                if missing_in_upload:
                    st.error(
                        "Your file is missing these required columns:\n\n"
                        + ", ".join(missing_in_upload)
                    )
                else:
                    # keep only the expected columns, in correct order
                    batch_X = batch_df[expected_cols].copy()

                    # 4. Run predictions
                    probs = home_best_model.predict_proba(batch_X)[:, 1]
                    preds = (probs >= 0.5).astype(int)

                    out = batch_df.copy()
                    out["default_probability"] = probs
                    out["predicted_default"] = preds

                    st.markdown("#### Prediction Summary")
                    st.write(
                        f"Number of rows: **{len(out)}**  \n"
                        f"Predicted defaults: **{int(out['predicted_default'].sum())}**"
                    )

                    st.dataframe(out.head())

                    # 5. Allow user to download the results
                    out_csv = out.to_csv(index=False)
                    st.download_button(
                        label="Download predictions as CSV",
                        data=out_csv,
                        file_name="batch_predictions.csv",
                        mime="text/csv",
                    )

                    if extra_in_upload:
                        st.info(
                            "Note: The following columns were in your file but not used by the model:\n\n"
                            + ", ".join(extra_in_upload)
                        )

            except Exception as e:
                st.error(f"Could not process uploaded file: {e}")
