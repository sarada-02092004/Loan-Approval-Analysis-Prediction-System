import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from database import Base, engine, SessionLocal
from models import DatasetRecord, ProcessedDatasetRecord, ModelRunRecord, PredictionLog
from preprocessing import validate_dataset, dataset_summary, missing_values_report, preprocess_loan_data
from eda import (
    basic_inspection, univariate_analysis, bivariate_analysis, multivariate_analysis,
    class_imbalance, outlier_analysis, correlation_analysis, eda_findings
)
from ml_model import train_and_compare_models, predict_from_payload, batch_predict
from schemas import (
    UploadResponse, DatasetSummaryResponse, MissingValuesResponse, PreprocessResponse,
    EDAResponse, TrainModelsResponse, PredictionRequest, PredictionResponse,
    BatchPredictionResponse, HistoryResponse
)
from utils import save_uploaded_file, load_dataset

app = FastAPI(
    title="Loan Approval Prediction API",
    description="Phase 1 + Phase 2 + Phase 3 + Phase 4 backend: upload, preprocess, EDA, model training, history, batch prediction, and downloads.",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

def get_dataset_or_404(db, dataset_id: int):
    dataset = db.query(DatasetRecord).filter(DatasetRecord.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset

@app.get("/")
def root():
    return {
        "message": "Loan Approval Prediction API is running.",
        "phase": "Phase 1 + Phase 2 + Phase 3 + Phase 4",
        "available_endpoints": [
            "/upload-data",
            "/summary/{dataset_id}",
            "/missing-values/{dataset_id}",
            "/preprocess/{dataset_id}",
            "/eda/basic/{dataset_id}",
            "/eda/univariate/{dataset_id}",
            "/eda/bivariate/{dataset_id}",
            "/eda/multivariate/{dataset_id}",
            "/eda/class-imbalance/{dataset_id}",
            "/eda/outliers/{dataset_id}",
            "/eda/correlation/{dataset_id}",
            "/eda/findings/{dataset_id}",
            "/train-models/{dataset_id}",
            "/predict/{dataset_id}",
            "/predict-batch/{dataset_id}",
            "/history/models/{dataset_id}",
            "/history/predictions/{dataset_id}",
            "/download/model/{dataset_id}",
            "/download/batch/{dataset_id}",
            "/download/processed/{dataset_id}",
        ],
    }

@app.post("/upload-data", response_model=UploadResponse)
async def upload_data(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    path = save_uploaded_file(file)
    try:
        df = load_dataset(str(path))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {exc}")

    is_valid, missing = validate_dataset(df)
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Dataset schema mismatch. Missing columns: {missing}")

    db = SessionLocal()
    try:
        record = DatasetRecord(
            filename=file.filename,
            row_count=int(df.shape[0]),
            column_count=int(df.shape[1]),
            columns_json=json.dumps(df.columns.tolist()),
            file_path=str(path)
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        return UploadResponse(
            message="Dataset uploaded successfully.",
            dataset_id=record.id,
            filename=record.filename,
            rows=record.row_count,
            columns=record.column_count
        )
    finally:
        db.close()

@app.get("/summary/{dataset_id}", response_model=DatasetSummaryResponse)
def get_summary(dataset_id: int):
    db = SessionLocal()
    try:
        dataset = get_dataset_or_404(db, dataset_id)
        df = load_dataset(dataset.file_path)
        summary = dataset_summary(df)
        return DatasetSummaryResponse(dataset_id=dataset_id, **summary)
    finally:
        db.close()

@app.get("/missing-values/{dataset_id}", response_model=MissingValuesResponse)
def get_missing_values(dataset_id: int):
    db = SessionLocal()
    try:
        dataset = get_dataset_or_404(db, dataset_id)
        df = load_dataset(dataset.file_path)
        report = missing_values_report(df)
        return MissingValuesResponse(dataset_id=dataset_id, **report)
    finally:
        db.close()

@app.post("/preprocess/{dataset_id}", response_model=PreprocessResponse)
def preprocess_dataset(dataset_id: int):
    db = SessionLocal()
    try:
        dataset = get_dataset_or_404(db, dataset_id)
        df = load_dataset(dataset.file_path)
        result = preprocess_loan_data(df)

        processed_dir = Path("data") / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        processed_path = processed_dir / f"processed_dataset_{dataset_id}.csv"
        result["dataframe"].to_csv(processed_path, index=False)

        processed_record = ProcessedDatasetRecord(
            source_dataset_id=dataset_id,
            row_count=result["processed_shape"]["rows"],
            column_count=result["processed_shape"]["columns"],
            file_path=str(processed_path),
            notes="Preprocessing completed with imputation, feature engineering, encoding, and Loan_ID removal."
        )
        db.add(processed_record)
        db.commit()
        db.refresh(processed_record)

        return PreprocessResponse(
            message="Preprocessing completed successfully.",
            processed_dataset_id=processed_record.id,
            original_shape=result["original_shape"],
            processed_shape=result["processed_shape"],
            engineered_features=result["engineered_features"],
            dropped_columns=result["dropped_columns"]
        )
    finally:
        db.close()

def _eda_response(dataset_id: int, fn):
    db = SessionLocal()
    try:
        dataset = get_dataset_or_404(db, dataset_id)
        df = load_dataset(dataset.file_path)
        return EDAResponse(dataset_id=dataset_id, data=fn(df))
    finally:
        db.close()

@app.get("/eda/basic/{dataset_id}", response_model=EDAResponse)
def eda_basic(dataset_id: int):
    return _eda_response(dataset_id, basic_inspection)

@app.get("/eda/univariate/{dataset_id}", response_model=EDAResponse)
def eda_univariate(dataset_id: int):
    return _eda_response(dataset_id, univariate_analysis)

@app.get("/eda/bivariate/{dataset_id}", response_model=EDAResponse)
def eda_bivariate(dataset_id: int):
    return _eda_response(dataset_id, bivariate_analysis)

@app.get("/eda/multivariate/{dataset_id}", response_model=EDAResponse)
def eda_multivariate(dataset_id: int):
    return _eda_response(dataset_id, multivariate_analysis)

@app.get("/eda/class-imbalance/{dataset_id}", response_model=EDAResponse)
def eda_class_imbalance(dataset_id: int):
    return _eda_response(dataset_id, class_imbalance)

@app.get("/eda/outliers/{dataset_id}", response_model=EDAResponse)
def eda_outliers(dataset_id: int):
    return _eda_response(dataset_id, outlier_analysis)

@app.get("/eda/correlation/{dataset_id}", response_model=EDAResponse)
def eda_correlation(dataset_id: int):
    return _eda_response(dataset_id, correlation_analysis)

@app.get("/eda/findings/{dataset_id}", response_model=EDAResponse)
def eda_findings_route(dataset_id: int):
    return _eda_response(dataset_id, eda_findings)

@app.post("/train-models/{dataset_id}", response_model=TrainModelsResponse)
def train_models(dataset_id: int):
    db = SessionLocal()
    try:
        dataset = get_dataset_or_404(db, dataset_id)
        df = load_dataset(dataset.file_path)
        result = train_and_compare_models(df, dataset_id)

        model_run = ModelRunRecord(
            source_dataset_id=dataset_id,
            best_model_name=result["best_model_name"],
            best_model_f1=result["metrics"][result["best_model_name"]]["f1_score"],
            best_model_accuracy=result["metrics"][result["best_model_name"]]["accuracy"],
            model_path=result["best_model_path"],
            metrics_json=json.dumps(result["metrics"])
        )
        db.add(model_run)
        db.commit()

        return TrainModelsResponse(
            dataset_id=dataset_id,
            best_model_name=result["best_model_name"],
            best_model_path=result["best_model_path"],
            trained_models=result["trained_models"],
            metrics=result["metrics"]
        )
    finally:
        db.close()

@app.post("/predict/{dataset_id}", response_model=PredictionResponse)
def predict(dataset_id: int, payload: PredictionRequest):
    db = SessionLocal()
    try:
        _ = get_dataset_or_404(db, dataset_id)
        result = predict_from_payload(dataset_id, payload.model_dump())

        pred_log = PredictionLog(
            source=f"dataset_{dataset_id}",
            prediction=result["prediction_label"],
            probability=result["probability_approved"],
            payload_json=json.dumps(payload.model_dump())
        )
        db.add(pred_log)
        db.commit()

        return PredictionResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        db.close()

@app.post("/predict-batch/{dataset_id}", response_model=BatchPredictionResponse)
async def predict_batch(dataset_id: int, file: UploadFile = File(...)):
    db = SessionLocal()
    try:
        _ = get_dataset_or_404(db, dataset_id)
        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are supported for batch prediction.")
        try:
            batch_df = pd.read_csv(file.file)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to read batch CSV: {exc}")

        result = batch_predict(dataset_id, batch_df)

        pred_log = PredictionLog(
            source=f"batch_dataset_{dataset_id}",
            prediction=f"BATCH:{result['total_rows']}",
            probability=None,
            payload_json=json.dumps({"file_name": file.filename, "rows": result["total_rows"], "output_file": result["output_file"]})
        )
        db.add(pred_log)
        db.commit()

        return BatchPredictionResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        db.close()

@app.get("/history/models/{dataset_id}", response_model=HistoryResponse)
def model_history(dataset_id: int):
    db = SessionLocal()
    try:
        _ = get_dataset_or_404(db, dataset_id)
        rows = db.query(ModelRunRecord).filter(ModelRunRecord.source_dataset_id == dataset_id).order_by(ModelRunRecord.created_at.desc()).all()
        records = [
            {
                "id": r.id,
                "best_model_name": r.best_model_name,
                "best_model_f1": r.best_model_f1,
                "best_model_accuracy": r.best_model_accuracy,
                "model_path": r.model_path,
                "created_at": r.created_at.isoformat()
            }
            for r in rows
        ]
        return HistoryResponse(dataset_id=dataset_id, records=records)
    finally:
        db.close()

@app.get("/history/predictions/{dataset_id}", response_model=HistoryResponse)
def prediction_history(dataset_id: int):
    db = SessionLocal()
    try:
        _ = get_dataset_or_404(db, dataset_id)
        prefix = f"dataset_{dataset_id}"
        batch_prefix = f"batch_dataset_{dataset_id}"
        rows = db.query(PredictionLog).filter(
            (PredictionLog.source == prefix) | (PredictionLog.source == batch_prefix)
        ).order_by(PredictionLog.created_at.desc()).all()
        records = [
            {
                "id": r.id,
                "source": r.source,
                "prediction": r.prediction,
                "probability": r.probability,
                "payload_json": r.payload_json,
                "created_at": r.created_at.isoformat()
            }
            for r in rows
        ]
        return HistoryResponse(dataset_id=dataset_id, records=records)
    finally:
        db.close()

@app.get("/download/model/{dataset_id}")
def download_model(dataset_id: int):
    model_path = Path("models") / f"best_model_dataset_{dataset_id}.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found.")
    return FileResponse(path=model_path, filename=model_path.name, media_type="application/octet-stream")

@app.get("/download/batch/{dataset_id}")
def download_batch(dataset_id: int):
    file_path = Path("data") / "batch_predictions" / f"batch_prediction_dataset_{dataset_id}.csv"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Batch prediction output not found.")
    return FileResponse(path=file_path, filename=file_path.name, media_type="text/csv")

@app.get("/download/processed/{dataset_id}")
def download_processed(dataset_id: int):
    file_path = Path("data") / "processed" / f"processed_dataset_{dataset_id}.csv"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Processed dataset not found.")
    return FileResponse(path=file_path, filename=file_path.name, media_type="text/csv")
