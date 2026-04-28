# src/app.py

import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

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
    roc_curve,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


# ------------------------------------------------------------------------------
#  CONFIG & STYLING
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="Credit Risk Intelligence Dashboard",
    page_icon="",
    layout="wide",
)

st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 1rem; }

.metric-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #16213e 100%);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    border-left: 4px solid #4fc3f7;
    margin-bottom: 0.5rem;
}
.metric-card h3 { color: #4fc3f7; font-size: 0.85rem; margin: 0; font-weight: 500; letter-spacing: 0.05em; }
.metric-card h1 { color: #ffffff; font-size: 2rem; margin: 0.2rem 0 0 0; font-weight: 700; }

.risk-high {
    background: linear-gradient(135deg, #7f1d1d, #450a0a);
    border-left: 4px solid #f87171;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.risk-medium {
    background: linear-gradient(135deg, #78350f, #451a03);
    border-left: 4px solid #fbbf24;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.risk-low {
    background: linear-gradient(135deg, #14532d, #052e16);
    border-left: 4px solid #4ade80;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.risk-label { font-size: 0.8rem; color: #94a3b8; letter-spacing: 0.1em; margin-bottom: 0.3rem; }
.risk-value { font-size: 3rem; font-weight: 800; color: white; line-height: 1; }
.risk-tag { font-size: 1rem; font-weight: 600; margin-top: 0.5rem; }

.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #4fc3f7;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}

.shap-box {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.2rem;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
#  DATA LOADING
# ------------------------------------------------------------------------------

@st.cache_data
def load_home_credit():
    data_path = Path(__file__).resolve().parents[1] / "data" / "home_credit_sample.csv"
    if not data_path.exists():
        return None, f"Could not find file at: {data_path}"
    df = pd.read_csv(data_path)
    return df, None


# ------------------------------------------------------------------------------
#  MODEL TRAINING
# ------------------------------------------------------------------------------

@st.cache_resource(show_spinner="Training models — hang tight...")
def train_home_models(df: pd.DataFrame):
    df = df.copy()
    if "TARGET" not in df.columns:
        raise ValueError("Expected column 'TARGET' in home_credit_sample.csv.")

    id_cols = [c for c in df.columns if c.upper().startswith("SK_ID")]
    df = df.drop(columns=id_cols, errors="ignore")

    y = df["TARGET"].astype(int)
    X = df.drop(columns=["TARGET"])

    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42,
        stratify=y if y.nunique() == 2 else None,
    )

    models = {}

    pipe_lr = Pipeline(steps=[
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe_lr.fit(X_train, y_train)
    models["Logistic Regression"] = pipe_lr

    pipe_rf = Pipeline(steps=[
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1,
            class_weight="balanced_subsample",
        )),
    ])
    pipe_rf.fit(X_train, y_train)
    models["Random Forest"] = pipe_rf

    if HAS_XGB:
        pipe_xgb = Pipeline(steps=[
            ("pre", preprocessor),
            ("clf", XGBClassifier(
                n_estimators=400, learning_rate=0.05, max_depth=4,
                subsample=0.8, colsample_bytree=0.8,
                objective="binary:logistic", eval_metric="logloss",
                n_jobs=-1, random_state=42,
            )),
        ])
        pipe_xgb.fit(X_train, y_train)
        models["XGBoost"] = pipe_xgb

    rows = []
    for name, model in models.items():
        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
        })

    metrics_df = pd.DataFrame(rows).set_index("model").sort_values("roc_auc", ascending=False)
    best_name = metrics_df["roc_auc"].idxmax()
    best_model = models[best_name]

    # Pre-compute SHAP values for best model on test set (sample for speed)
    shap_values = None
    shap_sample = None
    feature_names_out = None

    if HAS_SHAP:
        try:
            pre = best_model.named_steps["pre"]
            clf = best_model.named_steps["clf"]
            X_test_transformed = pre.transform(X_test)

            # Get feature names after preprocessing
            try:
                feature_names_out = pre.get_feature_names_out()
            except Exception:
                feature_names_out = [f"f{i}" for i in range(X_test_transformed.shape[1])]

            # Sample up to 300 rows for speed
            n_sample = min(300, X_test_transformed.shape[0])
            idx = np.random.RandomState(42).choice(X_test_transformed.shape[0], n_sample, replace=False)
            X_shap = X_test_transformed[idx]

            if isinstance(X_shap, np.ndarray) is False:
                X_shap = X_shap.toarray()

            explainer = shap.TreeExplainer(clf) if HAS_XGB and isinstance(clf, XGBClassifier) else shap.Explainer(clf, X_shap)
            shap_values = explainer.shap_values(X_shap)

            # For binary classifiers, shap_values may be a list [class0, class1]
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            shap_sample = X_shap

        except Exception as e:
            shap_values = None
            shap_sample = None

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
        "shap_values": shap_values,
        "shap_sample": shap_sample,
        "feature_names_out": feature_names_out,
    }


# ------------------------------------------------------------------------------
#  HELPER: SINGLE APPLICANT FEATURE SELECTION
# ------------------------------------------------------------------------------

def pick_single_input_features(df, target_col="TARGET"):
    df = df.drop(columns=[c for c in df.columns if c.upper().startswith("SK_ID")], errors="ignore")
    df_feat = df.drop(columns=[target_col], errors="ignore")

    num_cols_all = df_feat.select_dtypes(include="number").columns.tolist()
    cat_cols_all = df_feat.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    def top_by_non_null(cols, k):
        if not cols:
            return []
        counts = df_feat[cols].notna().sum().sort_values(ascending=False)
        return [c for c in counts.index[:k]]

    return top_by_non_null(num_cols_all, k=6), top_by_non_null(cat_cols_all, k=4)


# ------------------------------------------------------------------------------
#  SHAP WATERFALL FOR SINGLE PREDICTION
# ------------------------------------------------------------------------------

def compute_single_shap(home_result, df_input):
    """Compute SHAP values for a single input row."""
    if not HAS_SHAP:
        return None, None, None
    try:
        best_model = home_result["best_model"]
        pre = best_model.named_steps["pre"]
        clf = best_model.named_steps["clf"]

        X_transformed = pre.transform(df_input)
        if not isinstance(X_transformed, np.ndarray):
            X_transformed = X_transformed.toarray()

        try:
            feature_names = pre.get_feature_names_out()
        except Exception:
            feature_names = [f"f{i}" for i in range(X_transformed.shape[1])]

        if HAS_XGB and isinstance(clf, XGBClassifier):
            explainer = shap.TreeExplainer(clf)
        else:
            bg = home_result.get("shap_sample")
            explainer = shap.Explainer(clf, bg)

        sv = explainer.shap_values(X_transformed)
        if isinstance(sv, list):
            sv = sv[1]

        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[1] if len(base_val) > 1 else base_val[0]

        return sv[0], feature_names, float(base_val)
    except Exception as e:
        return None, None, None


def plot_waterfall(shap_vals, feature_names, base_value, final_prob, top_n=10):
    """Plot a clean waterfall chart using plotly."""
    df_shap = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_vals
    })
    df_shap["abs"] = df_shap["shap_value"].abs()
    df_shap = df_shap.nlargest(top_n, "abs").sort_values("shap_value")

    colors = ["#f87171" if v > 0 else "#4ade80" for v in df_shap["shap_value"]]

    fig = go.Figure(go.Bar(
        x=df_shap["shap_value"],
        y=df_shap["feature"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in df_shap["shap_value"]],
        textposition="outside",
    ))

    fig.update_layout(
        title=dict(
            text=f"SHAP Explanation — Why {final_prob:.1%} Default Probability?",
            font=dict(size=15, color="#e2e8f0")
        ),
        xaxis_title="SHAP Value (impact on prediction)",
        yaxis_title="",
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        xaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#4fc3f7", zerolinewidth=2),
        yaxis=dict(gridcolor="#1e3a5f"),
        height=420,
        margin=dict(l=10, r=80, t=50, b=40),
        showlegend=False,
    )

    # Add baseline annotation
    fig.add_vline(x=0, line_color="#4fc3f7", line_width=1.5)
    return fig


def plot_shap_summary(shap_values, feature_names, top_n=15):
    """Bar chart of global SHAP importance."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    df_imp = pd.DataFrame({"feature": feature_names, "importance": mean_abs})
    df_imp = df_imp.nlargest(top_n, "importance").sort_values("importance")

    fig = go.Figure(go.Bar(
        x=df_imp["importance"],
        y=df_imp["feature"],
        orientation="h",
        marker=dict(
            color=df_imp["importance"],
            colorscale="Blues",
            showscale=False,
        ),
        text=[f"{v:.3f}" for v in df_imp["importance"]],
        textposition="outside",
    ))

    fig.update_layout(
        title=dict(text="Global Feature Importance (Mean |SHAP|)", font=dict(size=15, color="#e2e8f0")),
        xaxis_title="Mean Absolute SHAP Value",
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        xaxis=dict(gridcolor="#1e3a5f"),
        yaxis=dict(gridcolor="#1e3a5f"),
        height=500,
        margin=dict(l=10, r=80, t=50, b=40),
    )
    return fig


# ------------------------------------------------------------------------------
#  LOAD DATA & TRAIN
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
#  HEADER
# ------------------------------------------------------------------------------

st.markdown("""
<div style="background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
            border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
            border-bottom: 3px solid #4fc3f7;">
    <h1 style="color: #ffffff; margin: 0; font-size: 2rem; font-weight: 800;">
         Credit Risk Intelligence Dashboard
    </h1>
    <p style="color: #94a3b8; margin: 0.5rem 0 0 0; font-size: 1rem;">
        End-to-end ML pipeline · XGBoost · SHAP Explainability · Home Credit Dataset
    </p>
</div>
""", unsafe_allow_html=True)

# Top-level metrics
if home_df is not None and home_result is not None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card"><h3>DATASET ROWS</h3><h1>{home_df.shape[0]:,}</h1></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card"><h3>FEATURES</h3><h1>{home_df.shape[1]-1}</h1></div>""", unsafe_allow_html=True)
    with c3:
        default_rate = home_df["TARGET"].mean() if "TARGET" in home_df.columns else 0
        st.markdown(f"""<div class="metric-card"><h3>DEFAULT RATE</h3><h1>{default_rate:.1%}</h1></div>""", unsafe_allow_html=True)
    with c4:
        best_auc = home_result["metrics"]["roc_auc"].max()
        st.markdown(f"""<div class="metric-card"><h3>BEST AUC-ROC</h3><h1>{best_auc:.3f}</h1></div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
#  TABS
# ------------------------------------------------------------------------------

tabs = st.tabs([
    "Overview",
    "Exploratory Analysis",
    "Model Performance",
    "Single Applicant Scoring",
    "Batch Scoring",
    "SHAP Explainability",
])


# ------------------------------------------------------------------------------
#  TAB 1 – OVERVIEW
# ------------------------------------------------------------------------------

with tabs[0]:
    st.markdown('<div class="section-header">Problem Statement</div>', unsafe_allow_html=True)
    st.markdown("""
> **"If I give this person a loan, how likely are they to default?"**

This dashboard answers that question with a full production-grade ML pipeline:
- **Individual scoring** — enter one applicant's details, get an instant risk score + explanation
- **Batch scoring** — upload a CSV, score thousands of applicants at once
- **Model comparison** — Logistic Regression vs Random Forest vs XGBoost
- **SHAP explainability** — understand *why* the model made every single decision
""")

    st.markdown('<div class="section-header">Pipeline Architecture</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    for col, step in zip(
        [col1, col2, col3, col4, col5],
        ["Raw Data", "Preprocessing", "Model Training", "Evaluation", "Explanation"],
    ):
        col.markdown(f"""
<div style="background:#1e3a5f; border-radius:10px; padding:1rem; text-align:center;">
    <div style="font-size:1.8rem">{icon}</div>
    <div style="color:#e2e8f0; font-weight:600; font-size:0.85rem; margin-top:0.3rem">{step}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Dataset Sample</div>', unsafe_allow_html=True)
    if home_df is not None:
        st.dataframe(home_df.head(), use_container_width=True)


# ------------------------------------------------------------------------------
#  TAB 2 – EDA
# ------------------------------------------------------------------------------

with tabs[1]:
    st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    if home_df is None:
        st.warning("No data available.")
    else:
        df_plot = home_df.drop(
            columns=[c for c in home_df.columns if c.upper().startswith("SK_ID")],
            errors="ignore"
        )

        st.markdown("#### Target Distribution")
        if "TARGET" in df_plot.columns:
            vc = df_plot["TARGET"].value_counts().reset_index()
            vc.columns = ["TARGET", "count"]
            vc["label"] = vc["TARGET"].map({0: "Non-Default ✅", 1: "Default ❌"})
            fig = px.bar(vc, x="label", y="count", color="label",
                        color_discrete_map={"Non-Default ✅": "#4ade80", "Default ❌": "#f87171"},
                        text="count", title="Class Distribution")
            fig.update_layout(plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                            font=dict(color="#e2e8f0"), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        num_cols_eda = [c for c in df_plot.select_dtypes(include="number").columns if c != "TARGET"]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Univariate Distribution")
            num_feature = st.selectbox("Choose a numeric feature", num_cols_eda, key="eda_num")
            fig = px.histogram(df_plot, x=num_feature, color="TARGET" if "TARGET" in df_plot.columns else None,
                             marginal="box", nbins=40, opacity=0.75,
                             title=f"Distribution of {num_feature}")
            fig.update_layout(plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font=dict(color="#e2e8f0"))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Bivariate vs Target")
            feat_bi = st.selectbox("Feature vs TARGET", num_cols_eda, key="eda_bi")
            fig = px.box(df_plot, x="TARGET", y=feat_bi, points="outliers",
                        color="TARGET",
                        color_discrete_map={0: "#4ade80", 1: "#f87171"},
                        title=f"{feat_bi} by Default Status")
            fig.update_layout(plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                            font=dict(color="#e2e8f0"), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Correlation Heatmap")
        top_num = df_plot[num_cols_eda].notna().sum().sort_values(ascending=False).index[:12]
        corr = df_plot[top_num].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                       title="Feature Correlation Matrix")
        fig.update_layout(plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font=dict(color="#e2e8f0"))
        st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------------------
#  TAB 3 – MODEL PERFORMANCE
# ------------------------------------------------------------------------------

with tabs[2]:
    st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)

    if home_result is None:
        st.warning("Models not available.")
        if train_err:
            st.code(train_err)
    else:
        metrics_df = home_result["metrics"]
        best_name = home_result["best_name"]

        st.success(f" Best model by ROC-AUC: **{best_name}**")

        # Metrics table with highlighting
        st.dataframe(metrics_df.style.highlight_max(axis=0, color="#14532d").format("{:.3f}"),
                    use_container_width=True)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### ROC-AUC Comparison")
            fig = px.bar(
                metrics_df.reset_index(),
                x="model", y="roc_auc",
                color="roc_auc",
                color_continuous_scale="Blues",
                text=metrics_df["roc_auc"].round(3).values,
                title="ROC-AUC by Model"
            )
            fig.update_layout(plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                            font=dict(color="#e2e8f0"), showlegend=False,
                            yaxis=dict(range=[0.5, 1.0]))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Confusion Matrix")
            model_name = st.selectbox("Select model", metrics_df.index.tolist())
            model = home_result["models"][model_name]
            y_pred = model.predict(home_result["X_test"])
            cm = confusion_matrix(home_result["y_test"], y_pred, labels=[0, 1])
            cm_df = pd.DataFrame(cm, index=["True: No Default", "True: Default"],
                                columns=["Pred: No Default", "Pred: Default"])
            fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues",
                          title=f"Confusion Matrix — {model_name}")
            fig.update_layout(plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                            font=dict(color="#e2e8f0"))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### ROC Curves (All Models)")
        fig_roc = go.Figure()
        colors_roc = {"Logistic Regression": "#4fc3f7", "Random Forest": "#4ade80", "XGBoost": "#f87171"}
        for name, model in home_result["models"].items():
            proba = model.predict_proba(home_result["X_test"])[:, 1]
            fpr, tpr, _ = roc_curve(home_result["y_test"], proba)
            auc_val = roc_auc_score(home_result["y_test"], proba)
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={auc_val:.3f})",
                                        line=dict(color=colors_roc.get(name, "#ffffff"), width=2)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random", line=dict(dash="dash", color="#475569")))
        fig_roc.update_layout(
            title="ROC Curves", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font=dict(color="#e2e8f0"),
            xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"), height=400
        )
        st.plotly_chart(fig_roc, use_container_width=True)

        with st.expander(" Full Classification Report"):
            y_pred_best = home_result["best_model"].predict(home_result["X_test"])
            st.text(classification_report(home_result["y_test"], y_pred_best, digits=3))


# ------------------------------------------------------------------------------
#  TAB 4 – SINGLE APPLICANT SCORING
# ------------------------------------------------------------------------------

with tabs[3]:
    st.markdown('<div class="section-header">Single Applicant Risk Scoring</div>', unsafe_allow_html=True)

    if home_result is None or home_df is None:
        st.warning("Models not available.")
    else:
        best_model = home_result["best_model"]
        feature_cols = home_result["feature_cols"]
        num_pick, cat_pick = pick_single_input_features(home_df)

        st.markdown("Fill in the applicant details below. All other features default to dataset medians.")

        with st.form("single_applicant_form"):
            col_left, col_right = st.columns(2)
            input_data = {}

            with col_left:
                st.markdown("**Numeric Features**")
                for col in num_pick:
                    series = home_df[col]
                    median_val = float(series.median()) if series.notna().any() else 0.0
                    min_val = float(series.min()) if series.notna().any() else median_val * 0.5
                    max_val = float(series.max()) if series.notna().any() else median_val * 1.5
                    val = st.number_input(col, value=median_val, min_value=min_val, max_value=max_val,
                                        step=max((max_val - min_val) / 1000.0, 1e-3))
                    input_data[col] = val

            with col_right:
                st.markdown("**Categorical Features**")
                for col in cat_pick:
                    series = home_df[col].dropna()
                    options = ["(missing)"] + (series.value_counts().index.tolist()[:20] if not series.empty else [])
                    choice = st.selectbox(col, options)
                    input_data[col] = None if choice == "(missing)" else choice

            submitted = st.form_submit_button("Score Applicant", use_container_width=True)

        if submitted:
            row = {}
            for col in feature_cols:
                if col in input_data:
                    row[col] = input_data[col]
                elif col in home_df.columns:
                    if home_df[col].dtype.kind in "bifc":
                        row[col] = float(home_df[col].median())
                    else:
                        mode_val = home_df[col].mode(dropna=True)
                        row[col] = mode_val.iloc[0] if not mode_val.empty else None
                else:
                    row[col] = None

            df_input = pd.DataFrame([row])
            prob = best_model.predict_proba(df_input)[:, 1][0]

            # Risk bucket
            if prob >= 0.6:
                risk_class = "risk-high"
                risk_label = "🔴 HIGH RISK"
                risk_color = "#f87171"
            elif prob >= 0.3:
                risk_class = "risk-medium"
                risk_label = "🟡 MEDIUM RISK"
                risk_color = "#fbbf24"
            else:
                risk_class = "risk-low"
                risk_label = "🟢 LOW RISK"
                risk_color = "#4ade80"

            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"""
<div class="{risk_class}">
    <div class="risk-label">PREDICTED DEFAULT PROBABILITY</div>
    <div class="risk-value">{prob:.1%}</div>
    <div class="risk-tag" style="color:{risk_color}">{risk_label}</div>
</div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # SHAP Waterfall for this prediction
            st.markdown('<div class="section-header"> Why this score? — SHAP Explanation</div>', unsafe_allow_html=True)

            if HAS_SHAP:
                with st.spinner("Computing SHAP values..."):
                    sv, feat_names, base_val = compute_single_shap(home_result, df_input)

                if sv is not None:
                    fig_wf = plot_waterfall(sv, feat_names, base_val, prob, top_n=10)
                    st.plotly_chart(fig_wf, use_container_width=True)

                    # Top 3 drivers in plain English
                    df_sv = pd.DataFrame({"feature": feat_names, "shap": sv})
                    df_sv["abs"] = df_sv["shap"].abs()
                    top3 = df_sv.nlargest(3, "abs")

                    st.markdown("**Top 3 drivers for this prediction:**")
                    for _, r in top3.iterrows():
                        direction = " increased" if r["shap"] > 0 else " decreased"
                        st.markdown(f"- **{r['feature']}** {direction} default risk by `{r['shap']:+.3f}`")
                else:
                    st.info("SHAP explanation unavailable for this prediction.")
            else:
                st.info("Install `shap` package to enable prediction explanations.")


# ------------------------------------------------------------------------------
#  TAB 5 – BATCH SCORING
# ------------------------------------------------------------------------------

with tabs[4]:
    st.markdown('<div class="section-header">Batch Scoring</div>', unsafe_allow_html=True)

    if home_df is None or home_result is None:
        st.warning("Models not available.")
    else:
        feature_cols = home_result["feature_cols"]
        template_df = home_df[feature_cols].head(5)
        template_csv = template_df.to_csv(index=False).encode("utf-8")

        st.markdown("""
1. Download the CSV template below
2. Fill in one row per applicant
3. Upload to get default probabilities for all applicants
""")
        st.download_button(" Download batch template", data=template_csv,
                          file_name="batch_template.csv", mime="text/csv")

        uploaded = st.file_uploader("Upload completed CSV", type="csv")
        if uploaded:
            try:
                batch_df = pd.read_csv(uploaded)
                missing = [c for c in feature_cols if c not in batch_df.columns]
                if missing:
                    st.error(f"Missing columns: {', '.join(missing)}")
                elif batch_df.shape[0] == 0:
                    st.warning("File has no rows. Please add applicant data.")
                else:
                    probs = home_result["best_model"].predict_proba(batch_df[feature_cols])[:, 1]
                    result_df = batch_df.copy()
                    result_df["default_probability"] = probs
                    result_df["risk_tier"] = pd.cut(probs, bins=[0, 0.3, 0.6, 1.0],
                                                   labels=["Low", "Medium", "High"])

                    st.markdown(f"**Scored {len(result_df):,} applicants**")

                    # Risk distribution
                    vc = result_df["risk_tier"].value_counts().reset_index()
                    fig = px.pie(vc, names="risk_tier", values="count",
                               color="risk_tier",
                               color_discrete_map={"Low": "#4ade80", "Medium": "#fbbf24", "High": "#f87171"},
                               title="Risk Tier Distribution")
                    fig.update_layout(paper_bgcolor="#0f172a", font=dict(color="#e2e8f0"))
                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(result_df.head(20), use_container_width=True)
                    st.download_button("⬇️ Download scored CSV",
                                      data=result_df.to_csv(index=False).encode("utf-8"),
                                      file_name="scored_applicants.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Error: {e}")


# ------------------------------------------------------------------------------
#  TAB 6 – SHAP EXPLAINABILITY (GLOBAL)
# ------------------------------------------------------------------------------

with tabs[5]:
    st.markdown('<div class="section-header"> SHAP Explainability — Global Model Insights</div>', unsafe_allow_html=True)

    if home_result is None:
        st.warning("Models not available.")
    elif not HAS_SHAP:
        st.warning("SHAP not installed. Run: `pip install shap`")
    elif home_result.get("shap_values") is None:
        st.warning("SHAP values could not be computed. Check model compatibility.")
    else:
        shap_values = home_result["shap_values"]
        feature_names = home_result["feature_names_out"]
        best_name = home_result["best_name"]

        st.markdown(f"Showing SHAP analysis for **{best_name}** — the best performing model.")

        st.markdown("""
> **What is SHAP?** SHAP (SHapley Additive exPlanations) fairly credits each feature for its contribution
> to a prediction — like splitting a restaurant bill based on what each person actually ate.
> Positive values push toward default. Negative values push away from default.
""")

        st.markdown("---")

        # Global importance bar chart
        st.markdown("#### Global Feature Importance (Mean |SHAP| across all test samples)")
        fig_global = plot_shap_summary(shap_values, feature_names, top_n=15)
        st.plotly_chart(fig_global, use_container_width=True)

        st.markdown("---")

        # SHAP distribution per feature (beeswarm-style using box plots)
        st.markdown("#### SHAP Value Distribution per Feature")
        mean_abs = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:10]
        top_features = [feature_names[i] for i in top_idx]
        top_shap = shap_values[:, top_idx]

        fig_dist = go.Figure()
        colors_dist = px.colors.sequential.Blues[3:]
        for i, (feat, vals) in enumerate(zip(top_features, top_shap.T)):
            fig_dist.add_trace(go.Box(
                y=vals, name=feat,
                marker_color=colors_dist[i % len(colors_dist)],
                boxmean=True,
            ))
        fig_dist.update_layout(
            title="SHAP Value Spread per Top Feature",
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"),
            yaxis=dict(gridcolor="#1e3a5f", title="SHAP Value"),
            xaxis=dict(tickangle=-30),
            height=450,
            showlegend=False,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("---")
        st.markdown("#### How to Read This")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
<div style="background:#1e3a5f; border-radius:10px; padding:1rem;">
<h4 style="color:#4ade80">🟢 Negative SHAP</h4>
<p style="color:#e2e8f0; font-size:0.9rem">Feature pushed prediction <b>away from default</b>. Good sign for the applicant.</p>
</div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""
<div style="background:#1e3a5f; border-radius:10px; padding:1rem;">
<h4 style="color:#f87171">🔴 Positive SHAP</h4>
<p style="color:#e2e8f0; font-size:0.9rem">Feature pushed prediction <b>toward default</b>. Risk factor for the applicant.</p>
</div>""", unsafe_allow_html=True)
        with col3:
            st.markdown("""
<div style="background:#1e3a5f; border-radius:10px; padding:1rem;">
<h4 style="color:#4fc3f7"> Bar Height</h4>
<p style="color:#e2e8f0; font-size:0.9rem">Average absolute impact across all applicants. Taller = more important globally.</p>
</div>""", unsafe_allow_html=True)
