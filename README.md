# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to churn, built on the Telco Customer Churn dataset. Includes an interactive Streamlit app for live, single-customer predictions.

**[Live demo →](#)** (https://churnpredictorapplication.streamlit.app/)

---

## Problem

Customer churn is one of the biggest threats to recurring revenue in subscription-based businesses. This project builds a model that flags customers at high risk of leaving, so a retention team can step in before they do — and surfaces which factors drive that risk in the first place.

## Dataset

- **Source:** Telco Customer Churn dataset (`Telco-Customer-Churn.csv`)
- **Size:** 7,043 customers × 20 features → cleaned to **7,021 rows** after removing 22 duplicates
- **Target variable:** `Churn` (Yes / No) — **26.4% churned, 73.6% retained** (imbalanced, handled with SMOTE during training)

## Key EDA Findings

**Contract type is the single strongest churn driver.** Month-to-month customers churn at **42.6%**, versus 11.3% for one-year and just 2.8% for two-year contracts.


**Churn is heavily front-loaded in the customer lifecycle.** New customers (0–12 months tenure) churn at **47.4%**, dropping steadily to **6.6%** by the 61–72 month mark — retention risk is highest right after signup.


Other notable patterns from the notebook:
- **Payment method matters:** electronic check payers churn at **45.1%**, versus 15–19% for automatic bank transfer, credit card, or mailed check.
- **Fiber optic internet customers churn more** (41.8%) than DSL (18.9%) or no-internet customers (7.2%).
- **Churners pay more on average:** \$74.60/month vs. \$61.34/month for retained customers.
- Six low-signal columns (`gender`, `Partner`, `Dependents`, `PhoneService`, `StreamingTV`, `TotalCharges`) were dropped before modeling based on EDA.

## Model

| | |
|---|---|
| Algorithm | **XGBoost Classifier**, tuned via `RandomizedSearchCV` (5-fold CV, scored on recall) |
| Class imbalance handling | **SMOTE** oversampling within the training pipeline |
| Features (14) | SeniorCitizen, tenure, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges |
| Train/test split | 80/20, stratified on `Churn` |
| **Test Accuracy** | **67.4%** |
| **Test Recall (Churn class)** | **88.2%** |
| **Test Precision (Churn class)** | 44% |
| **F1-score (Churn class)** | 0.59 |

> The model was deliberately tuned for **recall on the churn class** rather than raw accuracy. In a churn use case, missing an actual churner (false negative) is usually far more costly than flagging a loyal customer for outreach (false positive) — so this model catches **~88% of customers who actually churn**, at the cost of some over-flagging (44% precision). That trade-off is a deliberate business call, not an oversight — worth stating explicitly to an interviewer.

## What Drives Churn

Based on the EDA above, the clearest churn signals are: **month-to-month contracts, electronic check payments, fiber optic internet, and low tenure.** A natural next step (see below) is adding SHAP values so the live app can explain *why* it flagged a specific customer, not just that it did.

## Tech Stack

- **Python** — pandas, scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **Streamlit** — interactive prediction app

## Running Locally

```bash
git clone https://github.com/sahilkumarkesarwani/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
├── customer_churn_predictor.ipynb   # EDA + model training
├── app.py                           # Streamlit prediction app
├── df.pkl                           # preprocessed dataframe (feeds UI dropdowns)
├── model.pkl                        # trained XGBoost pipeline
├── Telco-Customer-Churn.csv         # raw dataset
└── requirements.txt
```
