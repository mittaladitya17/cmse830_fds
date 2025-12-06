# src/app.py

import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# Try to import XGBoost; if not available we just skip it
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# ------------------------------------------------------------------------------
#  BASIC CONFIG
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="Credit Risk Scoring Dashboard",
    page_icon="💳",
    layout="wide",
)

st.markdown(
    """
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
#  DATA LOADING
# ------------------------------------------------------------------------------

@st.cache_data
def load_home_credit():
    """
    Load the Home Credit *sample* dataset from the data folder.
    We assume the file is named 'home_credit_sample.csv' and
    contains a binary target column called 'TARGET'.
    """
    data_path = Path(__file__).resolve().parents[1] / "data" / "home_credit_sample.csv"
    if not data_path.exists():
        return None, f"Could not find file at: {data_path}"
    df = pd.read_csv(data_path)
    return df, None


# ------------------------------------------------------------------------------
#  MODEL TRAINING FUNCTION
# ------------------------------------------------------------------------------

@st.cache_resource(show_spinner=True)
def train_home_models(df: pd.DataFrame):
    """
    Train several models (LogReg, RandomForest, XGBoost if available)
    on the Home Credit sample dataset.

    Returns a dictionary with:
        - metrics: DataFrame of performance metrics
        - models: dict of trained pipelines
        - best_model: pipeline with best ROC AUC
        - best_name: name of best model
        - X_test, y_test: hold-out test set
        - feature_cols, num_cols, cat_cols: schema info
    """
    df = df.copy()

    if "TARGET" not in df.columns:
        raise ValueError("Expected column 'TARGET' in home_credit_sample.csv.")

    # Drop obvious ID columns if present
    id_cols = [c for c in df.columns if c.upper().startswith("SK_ID")]
    df = df.drop(columns=id_cols, errors="ignore")

    y = df["TARGET"].astype(int)
    X = df.drop(columns=["TARGET"])

    # Split by type
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    # Preprocessing pipelines with imputation (this fixes the NaN errors)
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y if y.nunique() == 2 else None,
    )

    models = {}

    # Logistic Regression
    pipe_lr = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    pipe_lr.fit(X_train, y_train)
    models["Logistic Regression"] = pipe_lr

    # Random Forest
    pipe_rf = Pipeline(
        steps=[
            ("pre", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )
    pipe_rf.fit(X_train, y_train)
    models["Random Forest"] = pipe_rf

    # XGBoost (if available)
    if HAS_XGB:
        pipe_xgb = Pipeline(
            steps=[
                ("pre", preprocessor),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=400,
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
        models["XGBoost"] = pipe_xgb

    # Compute metrics on test set
    rows = []
    for name, model in models.items():
        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds, zero_division=0),
                "recall": recall_score(y_test, preds, zero_division=0),
                "f1": f1_score(y_test, preds, zero_division=0),
                "roc_auc": roc_auc_score(y_test, proba),
            }
        )

    metrics_df = pd.DataFrame(rows).set_index("model").sort_values(
        "roc_auc", ascending=False
    )

    best_name = metrics_df["roc_auc"].idxmax()
    best_model = models[best_name]

    return {
        "metrics": metrics_df,
        "models": models,
        "best_model": best_model,
        "best_name": best_name,
        "X_test": X_test,
        "y_test": y_test,
        "feature_cols": X.columns.tolist(),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }


# ------------------------------------------------------------------------------
#  HELPER FOR SINGLE-APPLICANT FEATURE SELECTION
# ------------------------------------------------------------------------------

def pick_single_input_features(df: pd.DataFrame, target_col: str = "TARGET"):
    """
    Pick a reasonable subset of numeric and categorical features
    for the single-applicant form, so the UI is not overwhelming.

    Strategy:
      - Take top few numeric columns by non-null count
      - Take top few categorical columns by non-null count
    """
    df = df.drop(columns=[c for c in df.columns if c.upper().startswith("SK_ID")], errors="ignore")
    if target_col in df.columns:
        df_feat = df.drop(columns=[target_col])
    else:
        df_feat = df.copy()

    num_cols_all = df_feat.select_dtypes(include="number").columns.tolist()
    cat_cols_all = df_feat.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Helper to sort by non-null count
    def top_by_non_null(cols, k):
        if not cols:
            return []
        counts = df_feat[cols].notna().sum().sort_values(ascending=False)
        return [c for c in counts.index[:k]]

    num_pick = top_by_non_null(num_cols_all, k=6)
    cat_pick = top_by_non_null(cat_cols_all, k=4)

    return num_pick, cat_pick


# ------------------------------------------------------------------------------
#  LOAD DATA & TRAIN MODELS (ONCE)
# ------------------------------------------------------------------------------

home_df, load_err = load_home_credit()
home_result = None
train_err = None

if home_df is not None:
    try:
        home_result = train_home_models(home_df)
    except Exception as e:
        train_err = str(e)
else:
    train_err = load_err


# ------------------------------------------------------------------------------
#  UI LAYOUT
# ------------------------------------------------------------------------------

st.title("💳 Credit Risk Scoring Dashboard")

tabs = st.tabs(
    [
        "1️⃣ Project Overview",
        "2️⃣ Exploratory Analysis",
        "3️⃣ Model Performance",
        "4️⃣ Single Applicant Scoring",
        "5️⃣ Batch Scoring",
    ]
)

# ------------------------------------------------------------------------------
#  TAB 1 – PROJECT OVERVIEW
# ------------------------------------------------------------------------------

with tabs[0]:
    st.subheader("Project Overview")

    st.markdown(
        """
### Problem Statement  

Banks and lenders constantly need to answer the question:  
> **"If I give this person a loan, how likely are they to default?"**

This app simulates a **credit risk scoring system**:

- For individual users (single prediction)
- For groups of users (batch CSV scoring)
- With model comparison on a more realistic, higher-dimensional dataset (Home Credit sample)

---


    )


    st.markdown("---")

    st.markdown("### Dataset Summary (Home Credit Sample)")
    if home_df is None:
        st.error("Dataset could not be loaded. Check that `data/home_credit_sample.csv` exists.")
        if train_err:
            st.code(train_err)
    else:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Rows (sample)", f"{home_df.shape[0]:,}")
        with col_b:
            st.metric("Features", f"{home_df.shape[1] - 1:,}")  # minus TARGET
        with col_c:
            if "TARGET" in home_df.columns:
                default_rate = home_df["TARGET"].mean()
                st.metric("Default Rate (sample)", f"{default_rate:.1%}")
            else:
                st.metric("Default Rate", "N/A")

        st.markdown("#### Example Rows")
        st.dataframe(home_df.head(), use_container_width=True)

        st.markdown("#### Basic Statistics (Numeric Features)")
        num_desc = home_df.select_dtypes(include="number").describe().T
        st.dataframe(num_desc.round(2), use_container_width=True)

        st.markdown(
            """
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
**Target definition (`TARGET`):**  

- `0` – loan was **repaid** on time (non-default)  
- `1` – loan went into **default / serious delinquency**  

All models in this app are trained to predict this target.
"""
        )



# ------------------------------------------------------------------------------
#  TAB 2 – EXPLORATORY ANALYSIS
# ------------------------------------------------------------------------------

with tabs[1]:
    st.subheader("Exploratory Data Analysis (EDA)")

    if home_df is None:
        st.warning("No data available for EDA.")
    else:
        # Drop IDs
        df_plot = home_df.drop(
            columns=[c for c in home_df.columns if c.upper().startswith("SK_ID")],
            errors="ignore",
        )

        st.markdown("### 2.1 Target Distribution")
        if "TARGET" in df_plot.columns:
            fig = px.histogram(
                df_plot,
                x="TARGET",
                nbins=2,
                text_auto=True,
                title="Distribution of Target (0 = Non-default, 1 = Default)",
            )
            fig.update_layout(bargap=0.2)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No TARGET column found – cannot show target distribution.")

        st.markdown("---")
        st.markdown("### 2.2 Interactive Univariate Distribution")

        # Choose a numeric feature
        num_cols = df_plot.select_dtypes(include="number").columns.tolist()
        if "TARGET" in num_cols:
            num_cols.remove("TARGET")

        if num_cols:
            col1, col2 = st.columns(2)
            with col1:
                num_feature = st.selectbox(
                    "Choose a numeric feature", num_cols, key="eda_num_feature"
                )
            with col2:
                show_by_target = st.checkbox(
                    "Color by TARGET (0/1)", value=True, key="eda_color_by_target"
                )

            if show_by_target and "TARGET" in df_plot.columns:
                fig = px.histogram(
                    df_plot,
                    x=num_feature,
                    color="TARGET",
                    marginal="box",
                    nbins=40,
                    opacity=0.7,
                    title=f"Distribution of {num_feature} by TARGET",
                )
            else:
                fig = px.histogram(
                    df_plot,
                    x=num_feature,
                    nbins=40,
                    opacity=0.8,
                    title=f"Distribution of {num_feature}",
                )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric features available for univariate analysis.")

        st.markdown("---")
        st.markdown("### 2.3 Bivariate Relationship (Feature vs. Target)")

        if ("TARGET" in df_plot.columns) and num_cols:
            feat_bi = st.selectbox(
                "Choose a numeric feature for relationship with TARGET",
                num_cols,
                key="eda_bi_feature",
            )

            fig = px.box(
                df_plot,
                x="TARGET",
                y=feat_bi,
                points="all",
                title=f"{feat_bi} distribution by TARGET",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need TARGET and at least one numeric feature for bivariate analysis.")

        st.markdown("---")
        st.markdown("### 2.4 Correlation Heatmap (Numeric Subset)")

        # For large feature sets, just take top 12 numeric columns by non-null count
        num_cols_all = df_plot.select_dtypes(include="number").columns.tolist()
        if "TARGET" in num_cols_all:
            num_cols_all.remove("TARGET")

        if len(num_cols_all) > 0:
            non_null_counts = df_plot[num_cols_all].notna().sum().sort_values(ascending=False)
            top_num = non_null_counts.index[: min(12, len(non_null_counts))]
            corr = df_plot[top_num].corr()

            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Correlation Heatmap (Top Numeric Features)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric features for correlation heatmap.")


# ------------------------------------------------------------------------------
#  TAB 3 – MODEL PERFORMANCE
# ------------------------------------------------------------------------------

with tabs[2]:
    st.subheader("Model Performance (Home Credit Sample)")

    if home_result is None:
        st.warning("Models are not available – check dataset or training error.")
        if train_err:
            st.code(train_err)
    else:
        metrics_df = home_result["metrics"]
        best_name = home_result["best_name"]

        st.markdown(
            """
These models were trained on the Home Credit sample, using a **75% / 25% train–test split**.  
All numeric features are **median-imputed + standardized**, while categorical features are **imputed + one-hot encoded**.
"""
        )

        st.markdown("### 3.1 Summary Metrics (Higher ROC AUC is Better)")
        st.dataframe(metrics_df.round(3), use_container_width=True)

        st.success(f"Best model on this sample (by ROC AUC): **{best_name}**")

        st.markdown("---")
        st.markdown("### 3.2 Confusion Matrix for a Selected Model")

        model_name = st.selectbox(
            "Choose model to inspect", metrics_df.index.tolist(), index=0
        )
        model = home_result["models"][model_name]
        X_test = home_result["X_test"]
        y_test = home_result["y_test"]

        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        cm_df = pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])

        fig = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale="Blues",
            title=f"Confusion Matrix – {model_name}",
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show classification report"):
            report = classification_report(y_test, y_pred, digits=3)
            st.text(report)


# ------------------------------------------------------------------------------
#  TAB 4 – SINGLE APPLICANT SCORING
# ------------------------------------------------------------------------------

with tabs[3]:
    st.subheader("Single Applicant Scoring")

    if home_result is None or home_df is None:
        st.warning("Models not available – cannot score single applicants.")
    else:
        best_model = home_result["best_model"]
        feature_cols = home_result["feature_cols"]

        # Choose a small subset of features for easier UI
        num_pick, cat_pick = pick_single_input_features(home_df, target_col="TARGET")

        st.markdown(
            """
Fill in a few key fields below.  
For all other features, the app will automatically use typical/median values from the dataset.
"""
        )

        with st.form("single_applicant_form"):
            col_left, col_right = st.columns(2)

            input_data = {}

            # Numeric fields
            with col_left:
                st.markdown("#### Numeric Features")
                for col in num_pick:
                    series = home_df[col]
                    median_val = float(series.median()) if series.notna().any() else 0.0
                    min_val = float(series.min()) if series.notna().any() else median_val * 0.5
                    max_val = float(series.max()) if series.notna().any() else median_val * 1.5

                    val = st.number_input(
                        f"{col}",
                        value=median_val,
                        min_value=min_val,
                        max_value=max_val,
                        step=max((max_val - min_val) / 1000.0, 1e-3),
                    )
                    input_data[col] = val

            # Categorical fields
            with col_right:
                st.markdown("#### Categorical Features")
                for col in cat_pick:
                    series = home_df[col].dropna()
                    if series.empty:
                        options = ["(missing)"]
                        default_idx = 0
                    else:
                        unique_vals = series.value_counts().index.tolist()
                        options = unique_vals[:20]
                        options.insert(0, "(missing)")
                        default_idx = 0

                    choice = st.selectbox(f"{col}", options, index=default_idx)
                    input_data[col] = None if choice == "(missing)" else choice

            submitted = st.form_submit_button("Score Applicant")

        if submitted:
            # Build a full row with all training columns
            row = {}

            for col in feature_cols:
                if col in input_data:
                    row[col] = input_data[col]
                elif col in home_df.columns:
                    # Use median/mode from dataset
                    if home_df[col].dtype.kind in "bifc":  # numeric
                        row[col] = float(home_df[col].median())
                    else:
                        # categorical
                        mode_val = home_df[col].mode(dropna=True)
                        row[col] = mode_val.iloc[0] if not mode_val.empty else None
                else:
                    row[col] = None

            df_input = pd.DataFrame([row])

            prob = best_model.predict_proba(df_input)[:, 1][0]
            st.metric(
                "Predicted Default Probability",
                f"{prob:.2%}",
                help="Higher probability = higher estimated risk of default.",
            )

            st.markdown(
                """
**Interpretation (high level):**

- Values closer to **0%** → model believes this client looks similar to **good payers** in the data.  
- Values closer to **100%** → model believes this client looks similar to **high-risk defaulters**.
"""
            )


# ------------------------------------------------------------------------------
#  TAB 5 – BATCH SCORING
# ------------------------------------------------------------------------------

with tabs[4]:
    st.subheader("Batch Scoring (Upload a CSV of Applicants)")

    if home_result is None:
        st.warning("Models not available – cannot run batch scoring.")
    else:
        best_model = home_result["best_model"]
        feature_cols = home_result["feature_cols"]

        st.markdown(
            """
You can download a **template CSV**, fill it with multiple applicants,  
and upload it back to get predicted **default probabilities** for each row.
"""
        )

        # Template for download
        template_df = pd.DataFrame(columns=feature_cols)
        csv_bytes = template_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download batch template CSV",
            data=csv_bytes,
            file_name="home_credit_batch_template.csv",
            mime="text/csv",
        )

        st.markdown("---")

        uploaded = st.file_uploader("Upload a CSV file with the same columns", type=["csv"])

        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded)

                # Align columns
                missing_cols = [c for c in feature_cols if c not in batch_df.columns]
                extra_cols = [c for c in batch_df.columns if c not in feature_cols]

                if missing_cols:
                    st.warning(
                        f"The following required columns are missing and will be filled with defaults: {missing_cols}"
                    )
                if extra_cols:
                    st.info(
                        f"The following extra columns are present and will be ignored: {extra_cols}"
                    )

                # Keep only known features; add missing columns as NaN
                batch_df = batch_df.reindex(columns=feature_cols)

                probs = best_model.predict_proba(batch_df)[:, 1]
                result_df = batch_df.copy()
                result_df["pred_default_prob"] = probs

                st.markdown("### Preview of Scored Applicants")
                st.dataframe(result_df.head(), use_container_width=True)

                # Simple histogram of predicted risk
                fig = px.histogram(
                    result_df,
                    x="pred_default_prob",
                    nbins=30,
                    title="Distribution of Predicted Default Probabilities",
                )
                st.plotly_chart(fig, use_container_width=True)

                out_csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📤 Download results with predictions",
                    data=out_csv,
                    file_name="home_credit_batch_predictions.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error("Something went wrong while scoring your file.")
                st.code(str(e))
