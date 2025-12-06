💳 Credit Risk Scoring Dashboard
Machine Learning Application for Predicting Loan Default Risk
📌 Overview

This project is an interactive Streamlit web application designed to simulate a real-world credit risk assessment system, similar to those used by banks, fintech companies, and lending institutions.

Users can explore data, compare machine learning models, run predictions for a single applicant, and upload full CSV files to score many applicants at once.

🎯 Project Objectives
✔ Build an end-to-end ML pipeline for credit default prediction
✔ Compare multiple classification models
✔ Provide rich, interactive Exploratory Data Analysis (EDA)
✔ Allow both single and batch risk scoring
✔ Create an intuitive dashboard for stakeholders
📚 Datasets Used
1. German Credit Dataset (credit-g.csv)

Used for:

Single applicant scoring

Batch scoring demo

Simple baseline risk model

Description:

~1000 applicants

Well-structured categorical + numeric features

Binary target indicating good/bad credit risk

2. Home Credit Sample Dataset (home_credit_sample.csv)

A subset extracted from the massive Kaggle Home Credit Default Risk dataset.
Used for:

Full EDA

Multi-model training & comparison

Realistic credit-risk modeling

Why this dataset?

Higher dimensionality & complexity

Non-linear patterns

Missing values, socioeconomic variables, loan behavior signals

Mimics real financial industry data

Target variable:

TARGET = 1 → Default

TARGET = 0 → Repaid

🛠 Workflow Summary
1. Initial Data Analysis (IDA)

Inspect structure, datatypes, shapes

Handle missing values & anomalies

Remove ID columns (SK_ID_CURR, etc.)

Explore balance of target variable

2. Exploratory Data Analysis (EDA)

Interactive Streamlit visualizations including:

🔹 Target distribution
🔹 Univariate feature distributions
🔹 Bivariate feature vs target relationships
🔹 Correlation heatmaps
🔹 Dynamic histogram & boxplot visualizations

Users can interactively choose numeric features and compare them across default outcomes.

🤖 Machine Learning Models

We train three models on the Home Credit sample dataset:

1️⃣ Logistic Regression

Fast, interpretable baseline

Handles class imbalance (class_weight="balanced")

2️⃣ Random Forest Classifier

Captures non-linear relationships

Handles missingness and categorical expansion well

Robust and high-performing on tabular data

3️⃣ XGBoost (if available)

Powerful gradient boosting model

Often among top Kaggle competition solutions

Handles irregular patterns & interactions effectively

📈 Model Evaluation

Each model is compared on a 25% hold-out test set using:

Accuracy

Precision

Recall

F1-score

ROC AUC (primary metric)

The dashboard shows:

✔ Interactive model comparison table
✔ Confusion matrix visualization
✔ Full classification report
✔ Automatic selection of best model
👤 Single Applicant Scoring

Using the German Credit model:

User enters applicant information

Missing features are auto-filled with dataset medians/modes

Outputs predicted default probability

Provides interpretation guidance

📦 Batch Scoring

Upload a CSV file containing multiple customers and receive:

A full scored dataset

Default probability for every applicant

Downloadable results file

A pre-formatted template CSV is provided in the app.

🧰 Tech Stack
Component	Technology
Web App	Streamlit
ML Models	scikit-learn, XGBoost
Data Processing	Pandas, NumPy
Visualization	Plotly, Matplotlib
Pipeline Caching	Streamlit cache
Deployment	Streamlit Cloud
🔮 Future Enhancements

 Add SHAP-based model explainability

 Add feature importance dashboards

 Deploy enhanced version online

 Improve synthetic feature generation

 Add hyperparameter tuning (RandomizedSearchCV/GridSearchCV)

 Add scorecard-like risk banding

📌 Current Status

This is the final, full version of the CMSE 830 project, incorporating:

A richer dataset (Home Credit sample)

Multiple ML models

Full-featured Streamlit application

Interactive EDA

Batch & single scoring

Well-documented pipeline
