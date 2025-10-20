# Finance Risk Dashboard – Credit Scoring & Risk Prediction App

## Project Overview
This project is an interactive Streamlit web application that demonstrates how data science and machine learning can be used to predict credit risk for financial institutions. It simulates a real-world scenario where banks or lending companies must decide whether to approve or reject loan applications based on applicant data.

The dashboard allows users to:  
    -Explore and understand the dataset through interactive EDA (Exploratory Data Analysis).  
    -Upload CSV files to perform batch risk scoring for multiple customers.    
    -Enter customer details manually for a single prediction with risk probability.    
    -Understand key model metrics like accuracy, precision, recall, and ROC-AUC.    
    -This project is ongoing and will evolve into a full end-to-end credit risk scoring solution.

## Project Goals:  

- Build an end-to-end ML pipeline for credit risk classification.  
- Visualize and analyze financial data interactively.  
- Handle class imbalance (common in credit datasets) effectively.  
- Allow users to experiment and explore predictions in multiple ways.

### Dataset

We use the German Credit dataset (from OpenML), a widely used benchmark in financial ML tasks.    
    - Rows: 1000 applicants.   
    - Target: class – whether the applicant has good or bad credit risk.   
    - Features include:  
        - Personal info (age, employment status, etc.).   
        - Loan details (amount, duration, purpose).    
        - Financial history (savings, credit history, checking account balance)

### Workflow
1. Initial Data Analysis (IDA).
    - Loaded and inspected the dataset (.info(), .describe() etc.). 
    - Checked for missing values and data types.  
    - Explored class distribution to understand imbalance.

2. Exploratory Data Analysis (EDA):  
    - Interactive bar charts, heatmaps, pairplots, and boxplots.  
    - Feature-level distribution and correlation with the target variable.  
    - Bivariate scatter plots and category-based breakdowns.

3. Preprocessing & Modeling:  
    - Encoded categorical variables with OneHotEncoder.  
    - Scaled numeric variables with StandardScaler.  
    - Addressed class imbalance using class_weight='balanced' in logistic regression.  
    - Built a modular Pipeline for preprocessing + training.

4. Evaluation: Evaluated model performance using metrics:  
    - Accuracy. 
    - Precision. 
    - Recall.   
    - F1-score. 
    - ROC-AUC

### Feature	Description

- EDA Dashboard:	Explore the dataset interactively: class distribution, correlations, feature insights.  
- Batch Prediction:	Upload a CSV file of customers and get predicted risk + probability.  
- Single Prediction:	Input customer details manually and get an instant risk score.  
- Model Metrics:	See how well the ML model performs with precision, recall, F1, and AUC.

## Tech Stack
- Python – Core language. 
- Streamlit – Dashboard & Web App. 
- Scikit-learn – Data preprocessing and model building. 
- Pandas / NumPy – Data manipulation. 
- Matplotlib / Seaborn / Plotly – Data visualization. 
- Joblib – Model saving/loading

## Future Plans 

- [ ] Add advanced visualizations (SHAP, feature importance). 
- [ ] Deploy app on Streamlit Cloud or HuggingFace Spaces. 
- [ ] Integrate multiple models (Random Forest, XGBoost). 
- [ ] Add interpretability dashboard

## Project Status
This is the midterm version (~50% complete). Upcoming work includes deeper feature engineering, advanced visualization, and deployment.