# Loan-Approval-Analysis-Prediction-System
Loan Approval Prediction using Machine Learning | EDA, Feature Engineering, Random Forest, Streamlit App | B.Tech Final Year Project
# 🏦 Loan Dataset Analysis using Machine Learning

A complete end-to-end Machine Learning project that predicts whether a 
loan application will be **Approved or Rejected** based on applicant 
details, built as a B.Tech Final Year Project.

---

## 📌 Project Overview

Banks receive thousands of loan applications daily. Manual evaluation
is slow, inconsistent, and prone to bias. This project builds a 
data-driven classification system that automates loan approval prediction
using historical applicant data.

---

## 📊 Dataset

- **Source:** Analytics Vidhya Loan Prediction Dataset
- **Records:** 614 rows
- **Features:** 13 columns
- **Target Variable:** Loan_Status (Y = Approved, N = Not Approved)

---

## 🔬 Project Pipeline
Raw Data → Inspection → Missing Value Treatment → EDA →
Preprocessing → Feature Engineering → Model Training →
Evaluation → Best Model → Streamlit App

---

## ⚙️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Visualization |
| Scikit-learn | ML models |
| Streamlit | Web application |
| Pickle | Model serialization |
| Google Colab | Development environment |

---

## 🤖 Models Used

| Model | Accuracy | F1 Score | AUC |
|---|---|---|---|
| Logistic Regression | ~82% | ~88% | ~85% |
| Decision Tree | ~78% | ~85% | ~76% |
| **Random Forest** ✅ | **~80%** | **~88%** | **~88%** |

> ✅ **Random Forest** selected as best model based on highest AUC (88%)
> and most consistent 5-Fold Cross Validation results.

---

## 🧪 Key Features of the Project

- ✅ Complete EDA — Univariate, Bivariate, Multivariate Analysis
- ✅ Missing value imputation using mode & median strategies
- ✅ Feature Engineering — Total_Income, LoanAmount_to_Income, EMI_Burden
- ✅ Log transformation to handle skewed columns
- ✅ Comparison of 3 classification models
- ✅ Confusion Matrix, ROC Curve, Cross Validation
- ✅ Feature Importance Analysis
- ✅ Streamlit web app for real-time prediction

---

## 🚀 How to Run the Streamlit App

### Step 1 — Clone the repository
```bash
git clone https://github.com/your-username/loan-dataset-analysis-ml.git
cd loan-dataset-analysis-ml
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the app
```bash
streamlit run app.py
```

### Step 4 — Open in browser
http://localhost:8501
---

## 📁 Project Structure
loan-dataset-analysis-ml/
│
├── app.py                  # Streamlit application
├── loan_model.pkl          # Trained Random Forest model
├── model_columns.pkl       # Feature column names
├── loan_data.csv           # Dataset
├── Loan_Analysis.ipynb     # Complete Jupyter notebook
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
---

## 📦 requirements.txt
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
---

## 🏆 Results

- **Best Model:** Random Forest Classifier
- **AUC Score:** 88%
- **Top Predictor:** Credit History
- **Insight:** Applicants with good credit history have ~80% approval rate

---

## 👤 Author

**Sarada Prasad Sahoo**
B.Tech — Computer Science & Engineering
Ajay Binay Institute of Technology, Cuttack
Regd. No.: 2201206052
Guide: Mr. Preetam Kumar Behera | Skill O Tech

---

## 📜 License

This project is submitted as a B.Tech Final Year Academic Project.
