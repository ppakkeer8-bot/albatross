Credit Risk Analysis, Prediction & Explainability
A machine learning project for predicting loan defaults with validation techniques and explainability using SHAP & LIME.

Project Overview
This project builds a credit risk prediction model using machine learning to classify loan applicants as defaulters or non-defaulters. The model is validated using multiple validation techniques and interpreted using SHAP & LIME to explain predictions.
Dataset
The dataset contains loan applicant information with features such as:

Personal details (Age, Income, Employment Length)

Loan details (Amount, Interest Rate, Grade, Intent)

Credit history (Default on File, Credit History Length)

Loan status (0 = No Default, 1 = Default)

Dataset Source: Public credit risk datasets from Kaggle / Lending Club / Home Credit.
Methodology
Data Preprocessing

Handling missing values

Encoding categorical variables

Normalization of numerical features

Model Training

Ensemble models: XGBoost / LightGBM

Hyperparameter tuning with GridSearchCV
Validation Techniques

K-Fold Cross Validation

Stratified K-Fold

Monte Carlo Simulation

Bootstrapping

Explainability

Global Explanations: SHAP summary plots for top features

Local Explanations: LIME & SHAP for individual loan cases

Results
Performance Metrics:

AUC: ~0.85

F1-Score: ~0.78

Top Global Features (SHAP):

Loan-to-Income Ratio

Loan Grade

Home Ownership

Income

Interest Rate
Validation Comparison: Accuracy across methods was consistent (~82–85%).

Local Case Studies:

Low-risk applicant → correctly predicted No Default

High-risk applicant → correctly predicted Default

Borderline applicant → mixed signals, explained via SHAP & LIME

Visuals
SHAP Summary Plot (Global Feature Importance)

LIME Explanation (Local Case Studies)

Validation Accuracy Comparison Chart

All plots are saved in the /plots folder.
Deliverables
Trained model parameters & metrics

Python code implementation (.py / Jupyter Notebook)

Executive summary (≤ 500 words)

Comparative analysis of SHAP vs LIME explanations

How to Run
# Clone the repository
git clone https://github.com/username/credit-risk-analysis.git

# Navigate to project folder
cd credit-risk-analysis

# Install dependencies
pip install -r requirements.txt

# Run Jupyter Notebook
jupyter notebook Credit_Risk_Analysis.ipynb

References
SHAP Documentation

LIME Paper (Ribeiro et al., 2016)

XGBoost Documentation

Credit Risk Datasets on Kaggle
