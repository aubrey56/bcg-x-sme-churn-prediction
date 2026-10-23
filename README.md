# 📊 PowerCo SME Customer Churn Prediction & Retention Analysis

> **BCG X Data Science Simulation Project**  
> An end-to-end machine learning pipeline and financial evaluation designed to predict customer churn, quantify price sensitivity, and optimize targeted retention strategies for PowerCo's SME (Small & Medium Enterprises) division.

---

## 📌 Project Overview

PowerCo, a major energy utility provider, is facing declining profitability due to customer churn in the SME market segment. This project investigates whether customer churn is primarily driven by **price sensitivity** and builds a predictive machine learning model to identify high-risk accounts before contract expiration.

### 🎯 Key Objectives
1. **Exploratory Data Analysis (EDA):** Identify primary drivers of customer churn across consumption patterns, contract tenure, and pricing metrics.
2. **Feature Engineering:** Calculate year-over-year price volatility, multi-period price differences across off-peak/peak hours, and time-based tenure features.
3. **Predictive Modeling:** Train and evaluate a **Random Forest Classifier** to accurately flag at-risk accounts.
4. **Business & Financial Impact Analysis:** Quantify the financial viability of offering a **20% targeted discount** strategy to retain high-margin customers.

---

## 💡 Key Insights & Findings

* **Baseline Churn Rate:** **9.71%** of SME customers churn annually.
* **Price Sensitivity Analysis:** Price sensitivity is **not** the primary driver of churn. Feature importance ranking reveals that historical consumption volume (`cons_12m`), net margin (`net_margin`), and overall customer tenure carry significantly higher predictive weight than price differences.
* **Financial Impact:** Offering blanket discounts to all customers severely damages operating margins. However, applying a **targeted 20% discount strictly to model-predicted high-risk churners** preserves **€4.8M+ in net annual margin**.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.8+
* **Data Processing & Analysis:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn` (`RandomForestClassifier`, `train_test_split`, `metrics`)

---

## 📈 Model Performance & Evaluation

Because customer churn is highly imbalanced ($\approx 10\%$ churn vs. $90\%$ non-churn), standard accuracy is misleading (a naive model predicting 100% non-churn achieves 90% accuracy). The model was evaluated using threshold-aware metrics:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 90.36% |
| **Precision** | 85.00% |
| **Recall (Churn Class)** | 4.64% (Baseline) |
| **ROC-AUC** | 0.7040 |

> **Note on Optimization:** To maximize business value, probability decision thresholds can be tuned (or `class_weight='balanced'` applied) to boost recall and capture more at-risk SME accounts prior to renewal.

---

## 📂 Project Structure

```text
├── data/
│   ├── clean_data_after_eda.csv   # Cleaned dataset post-EDA
│   └── data_for_predictions.csv   # Final dataset with engineered features
├── notebooks/
│   ├── 01_EDA.ipynb               # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb # Feature creation & encoding
│   └── 03_Modeling_Evaluation.ipynb # Random Forest training & evaluation
├── reports/
│   └── powerco_executive_summary.pdf # Steering Committee Executive Abstract Slide
├── README.md                      # Project documentation
└── requirements.txt               # Dependencies list
