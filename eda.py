import numpy as np
import pandas as pd

CATEGORICAL_COLUMNS = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Property_Area", "Loan_Status"
]

NUMERICAL_COLUMNS = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History"
]

def _safe_value_counts(series: pd.Series):
    return series.astype(str).fillna("Missing").value_counts(dropna=False).to_dict()

def basic_inspection(df: pd.DataFrame):
    return {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": df.columns.tolist(),
        "head": df.head(10).replace({np.nan: None}).to_dict(orient="records"),
        "tail": df.tail(10).replace({np.nan: None}).to_dict(orient="records"),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "describe_numeric": df.describe(include=[np.number]).fillna(0).round(3).to_dict(),
    }

def univariate_analysis(df: pd.DataFrame):
    categorical = {}
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            categorical[col] = _safe_value_counts(df[col])

    numerical = {}
    for col in NUMERICAL_COLUMNS:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            numerical[col] = {
                "count": int(s.count()),
                "mean": float(s.mean()) if s.count() else 0.0,
                "median": float(s.median()) if s.count() else 0.0,
                "std": float(s.std()) if s.count() else 0.0,
                "min": float(s.min()) if s.count() else 0.0,
                "max": float(s.max()) if s.count() else 0.0,
                "skew": float(s.skew()) if s.count() else 0.0,
                "histogram_bins": [float(x) for x in s.dropna().tolist()],
            }

    return {"categorical": categorical, "numerical": numerical}

def bivariate_analysis(df: pd.DataFrame):
    results = {}

    if "Loan_Status" in df.columns:
        for col in ["Credit_History", "Education", "Property_Area", "Married", "Self_Employed"]:
            if col in df.columns:
                pivot = pd.crosstab(df[col].astype(str), df["Loan_Status"].astype(str))
                results[f"{col}_vs_Loan_Status"] = pivot.to_dict()

        for col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]:
            if col in df.columns:
                grouped = df.groupby("Loan_Status")[col].mean(numeric_only=True).fillna(0).round(3).to_dict()
                results[f"{col}_mean_by_Loan_Status"] = {str(k): float(v) for k, v in grouped.items()}

    if "ApplicantIncome" in df.columns and "LoanAmount" in df.columns:
        tmp = df[["ApplicantIncome", "LoanAmount"]].copy()
        tmp["ApplicantIncome"] = pd.to_numeric(tmp["ApplicantIncome"], errors="coerce")
        tmp["LoanAmount"] = pd.to_numeric(tmp["LoanAmount"], errors="coerce")
        tmp = tmp.dropna()
        results["income_vs_loanamount_scatter"] = tmp.head(500).to_dict(orient="records")

    return results

def multivariate_analysis(df: pd.DataFrame):
    derived = df.copy()

    if {"ApplicantIncome", "CoapplicantIncome"}.issubset(df.columns):
        derived["TotalIncome"] = pd.to_numeric(derived["ApplicantIncome"], errors="coerce").fillna(0) + pd.to_numeric(derived["CoapplicantIncome"], errors="coerce").fillna(0)

    views = {}

    if {"TotalIncome", "LoanAmount", "Credit_History", "Loan_Status"}.issubset(derived.columns):
        subset = derived[["TotalIncome", "LoanAmount", "Credit_History", "Loan_Status"]].copy()
        subset["LoanAmount"] = pd.to_numeric(subset["LoanAmount"], errors="coerce")
        subset["Credit_History"] = pd.to_numeric(subset["Credit_History"], errors="coerce")
        subset = subset.dropna().head(500)
        views["income_loan_credit_vs_approval"] = subset.to_dict(orient="records")

    if {"Dependents", "Married", "ApplicantIncome", "Loan_Status"}.issubset(derived.columns):
        subset = derived[["Dependents", "Married", "ApplicantIncome", "Loan_Status"]].copy()
        subset["ApplicantIncome"] = pd.to_numeric(subset["ApplicantIncome"], errors="coerce")
        subset = subset.dropna().head(500)
        views["dependents_married_income_vs_approval"] = subset.to_dict(orient="records")

    return views

def class_imbalance(df: pd.DataFrame):
    if "Loan_Status" not in df.columns:
        return {"counts": {}, "percentages": {}}
    counts = df["Loan_Status"].astype(str).value_counts(dropna=False)
    percentages = (counts / len(df) * 100).round(2)
    return {
        "counts": counts.to_dict(),
        "percentages": percentages.to_dict()
    }

def outlier_analysis(df: pd.DataFrame):
    result = {}
    for col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) == 0:
                continue
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = int(((s < lower) | (s > upper)).sum())
            result[col] = {
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "outlier_count": outliers,
                "values": [float(x) for x in s.tolist()]
            }
    return result

def correlation_analysis(df: pd.DataFrame):
    cols = [c for c in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"] if c in df.columns]
    if not cols:
        return {"matrix": {}}
    num = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = num.corr().fillna(0).round(3)
    return {"matrix": corr.to_dict()}

def eda_findings(df: pd.DataFrame):
    findings = []

    if "Credit_History" in df.columns and "Loan_Status" in df.columns:
        ct = pd.crosstab(df["Credit_History"], df["Loan_Status"])
        findings.append("Credit history shows a strong visible relationship with loan approval patterns.")

    if "ApplicantIncome" in df.columns:
        skew = pd.to_numeric(df["ApplicantIncome"], errors="coerce").skew()
        findings.append(f"ApplicantIncome appears {'highly skewed' if abs(skew) > 1 else 'moderately distributed'} with skewness {round(float(skew), 3)}.")

    if "LoanAmount" in df.columns:
        skew = pd.to_numeric(df["LoanAmount"], errors="coerce").skew()
        findings.append(f"LoanAmount shows {'noticeable skewness' if abs(skew) > 1 else 'limited skewness'} with skewness {round(float(skew), 3)}.")

    if "Loan_Status" in df.columns:
        dist = df["Loan_Status"].value_counts(normalize=True) * 100
        if len(dist):
            top_class = dist.idxmax()
            findings.append(f"The target class is led by '{top_class}' at {round(float(dist.max()),2)}% of records.")

    return {"findings": findings}
