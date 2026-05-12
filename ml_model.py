from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)

from preprocessing import preprocess_loan_data

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def _metrics_dict(y_true, y_pred, y_proba=None):
    result = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }
    if y_proba is not None:
        try:
            result["roc_auc"] = round(float(roc_auc_score(y_true, y_proba)), 4)
        except Exception:
            result["roc_auc"] = None
    else:
        result["roc_auc"] = None
    return result

def prepare_features(df: pd.DataFrame):
    processed = preprocess_loan_data(df)["dataframe"].copy()
    if "Loan_Status" not in processed.columns:
        raise ValueError("Loan_Status column not found after preprocessing.")
    X = processed.drop(columns=["Loan_Status"])
    y = processed["Loan_Status"].astype(int)
    return X, y, processed.columns.tolist()

def train_and_compare_models(df: pd.DataFrame, dataset_id: int):
    X, y, final_columns = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, solver="liblinear", random_state=42))
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, min_samples_split=4
        )
    }

    metrics = {}
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model

        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]

        cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1")
        metrics[name] = _metrics_dict(y_test, y_pred, y_proba)
        metrics[name]["cross_validation_f1_mean"] = round(float(np.mean(cv_scores)), 4)
        metrics[name]["cross_validation_f1_std"] = round(float(np.std(cv_scores)), 4)

        if name in ["Decision Tree", "Random Forest"] and hasattr(model, "feature_importances_"):
            fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
            metrics[name]["feature_importance"] = {k: round(float(v), 6) for k, v in fi.to_dict().items()}
        else:
            metrics[name]["feature_importance"] = {}

    best_model_name = sorted(
        metrics.keys(),
        key=lambda m: (metrics[m]["f1_score"], metrics[m]["accuracy"], metrics[m]["cross_validation_f1_mean"]),
        reverse=True
    )[0]
    best_model = fitted_models[best_model_name]

    bundle = {
        "model": best_model,
        "best_model_name": best_model_name,
        "feature_columns": X.columns.tolist(),
        "all_metrics": metrics,
        "dataset_id": dataset_id,
    }
    model_path = MODEL_DIR / f"best_model_dataset_{dataset_id}.joblib"
    joblib.dump(bundle, model_path)

    return {
        "dataset_id": dataset_id,
        "best_model_name": best_model_name,
        "best_model_path": str(model_path),
        "trained_models": list(models.keys()),
        "metrics": metrics,
        "feature_columns": X.columns.tolist(),
        "final_columns": final_columns,
    }

def preprocess_single_input(payload: dict):
    df = pd.DataFrame([{
        "Loan_ID": "NEW_INPUT",
        "Gender": payload["Gender"],
        "Married": payload["Married"],
        "Dependents": payload["Dependents"],
        "Education": payload["Education"],
        "Self_Employed": payload["Self_Employed"],
        "ApplicantIncome": payload["ApplicantIncome"],
        "CoapplicantIncome": payload["CoapplicantIncome"],
        "LoanAmount": payload["LoanAmount"],
        "Loan_Amount_Term": payload["Loan_Amount_Term"],
        "Credit_History": payload["Credit_History"],
        "Property_Area": payload["Property_Area"],
        "Loan_Status": "Y",
    }])
    processed = preprocess_loan_data(df)["dataframe"]
    if "Loan_Status" in processed.columns:
        processed = processed.drop(columns=["Loan_Status"])
    return processed

def load_model_bundle(dataset_id: int):
    model_path = MODEL_DIR / f"best_model_dataset_{dataset_id}.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Best model not found. Train the models first.")
    return joblib.load(model_path)

def predict_from_payload(dataset_id: int, payload: dict):
    bundle = load_model_bundle(dataset_id)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    processed = preprocess_single_input(payload)

    for col in feature_columns:
        if col not in processed.columns:
            processed[col] = 0

    processed = processed[feature_columns]

    pred = int(model.predict(processed)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(processed)[0][1])

    return {
        "dataset_id": dataset_id,
        "best_model_name": bundle["best_model_name"],
        "prediction_label": "Approved" if pred == 1 else "Rejected",
        "prediction_numeric": pred,
        "probability_approved": round(proba, 4) if proba is not None else None,
        "processed_input": processed.iloc[0].to_dict()
    }

def batch_predict(dataset_id: int, df: pd.DataFrame):
    bundle = load_model_bundle(dataset_id)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    working = df.copy()
    if "Loan_Status" not in working.columns:
        working["Loan_Status"] = "Y"

    processed = preprocess_loan_data(working)["dataframe"].copy()
    if "Loan_Status" in processed.columns:
        processed = processed.drop(columns=["Loan_Status"])

    for col in feature_columns:
        if col not in processed.columns:
            processed[col] = 0
    processed = processed[feature_columns]

    preds = model.predict(processed)
    probs = model.predict_proba(processed)[:, 1] if hasattr(model, "predict_proba") else np.full(len(processed), np.nan)

    output = df.copy()
    output["Prediction_Numeric"] = preds
    output["Prediction_Label"] = np.where(output["Prediction_Numeric"] == 1, "Approved", "Rejected")
    output["Approval_Probability"] = probs

    batch_dir = Path("data") / "batch_predictions"
    batch_dir.mkdir(parents=True, exist_ok=True)
    out_path = batch_dir / f"batch_prediction_dataset_{dataset_id}.csv"
    output.to_csv(out_path, index=False)

    return {
        "dataset_id": dataset_id,
        "best_model_name": bundle["best_model_name"],
        "total_rows": int(len(output)),
        "output_file": str(out_path),
        "preview": output.head(20).replace({np.nan: None}).to_dict(orient="records")
    }
