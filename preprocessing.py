import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from utils import safe_mode

EXPECTED_COLUMNS = [
    "Loan_ID", "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "Credit_History", "Property_Area", "Loan_Status"
]

CATEGORICAL_COLUMNS = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Property_Area", "Loan_Status"
]

NUMERICAL_COLUMNS = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History"
]

def validate_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    return (len(missing) == 0, missing)

def dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_summary = df.describe(include=[np.number]).fillna(0).round(3).to_dict()
    target_dist = {}
    if "Loan_Status" in df.columns:
        target_dist = df["Loan_Status"].astype(str).value_counts(dropna=False).to_dict()

    return {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "target_distribution": target_dist,
        "numerical_summary": numeric_summary,
    }

def missing_values_report(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    missing_counts = df.isna().sum().to_dict()
    percentages = ((df.isna().mean() * 100).round(2)).to_dict()
    return {"missing_counts": missing_counts, "missing_percentages": percentages}

def preprocess_loan_data(df: pd.DataFrame) -> Dict[str, Any]:
    working = df.copy()

    # Normalize key categorical representations
    working["Dependents"] = working["Dependents"].replace("3+", 3)
    working["Dependents"] = pd.to_numeric(working["Dependents"], errors="coerce")

    # Missing value imputation
    for col in ["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Loan_Status"]:
        if col in working.columns:
            fill_value = safe_mode(working[col])
            working[col] = working[col].fillna(fill_value)

    if "Dependents" in working.columns:
        dep_mode = safe_mode(working["Dependents"])
        working["Dependents"] = working["Dependents"].fillna(dep_mode)

    for col in ["LoanAmount", "Loan_Amount_Term", "Credit_History", "ApplicantIncome", "CoapplicantIncome"]:
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")
            working[col] = working[col].fillna(working[col].median())

    # Feature engineering
    working["TotalIncome"] = working["ApplicantIncome"] + working["CoapplicantIncome"]
    working["Dependents"] = working["Dependents"].replace(0, 0)

    working["LoanAmount_to_Income"] = np.where(
        working["TotalIncome"] > 0,
        working["LoanAmount"] / working["TotalIncome"],
        0
    )

    working["Income_per_Dependent"] = working["TotalIncome"] / (working["Dependents"].fillna(0) + 1)

    working["Has_Coapplicant"] = np.where(working["CoapplicantIncome"] > 0, 1, 0)

    # Binary mappings for app-ready cleaned dataset
    binary_maps = {
        "Gender": {"Male": 1, "Female": 0},
        "Married": {"Yes": 1, "No": 0},
        "Education": {"Graduate": 1, "Not Graduate": 0},
        "Self_Employed": {"Yes": 1, "No": 0},
        "Loan_Status": {"Y": 1, "N": 0},
    }
    for col, mapper in binary_maps.items():
        if col in working.columns:
            working[col] = working[col].map(mapper)

    if "Property_Area" in working.columns:
        area_dummies = pd.get_dummies(working["Property_Area"], prefix="Property_Area", dtype=int)
        working = pd.concat([working.drop(columns=["Property_Area"]), area_dummies], axis=1)

    dropped_columns = []
    if "Loan_ID" in working.columns:
        working = working.drop(columns=["Loan_ID"])
        dropped_columns.append("Loan_ID")

    # Ensure final numeric dataset for downstream modeling
    working["Dependents"] = pd.to_numeric(working["Dependents"], errors="coerce").fillna(0)

    final_cols = working.columns.tolist()
    return {
        "dataframe": working,
        "original_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "processed_shape": {"rows": int(working.shape[0]), "columns": int(working.shape[1])},
        "engineered_features": [
            "TotalIncome", "LoanAmount_to_Income", "Income_per_Dependent", "Has_Coapplicant"
        ],
        "dropped_columns": dropped_columns,
        "final_columns": final_cols,
    }
