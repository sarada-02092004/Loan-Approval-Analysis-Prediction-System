from pydantic import BaseModel
from typing import Dict, List, Any, Optional

class UploadResponse(BaseModel):
    message: str
    dataset_id: int
    filename: str
    rows: int
    columns: int

class DatasetSummaryResponse(BaseModel):
    dataset_id: int
    shape: Dict[str, int]
    columns: List[str]
    dtypes: Dict[str, str]
    target_distribution: Dict[str, int]
    numerical_summary: Dict[str, Dict[str, Any]]

class MissingValuesResponse(BaseModel):
    dataset_id: int
    missing_counts: Dict[str, int]
    missing_percentages: Dict[str, float]

class PreprocessResponse(BaseModel):
    message: str
    processed_dataset_id: int
    original_shape: Dict[str, int]
    processed_shape: Dict[str, int]
    engineered_features: List[str]
    dropped_columns: List[str]

class EDAResponse(BaseModel):
    dataset_id: int
    data: Dict[str, Any]

class TrainModelsResponse(BaseModel):
    dataset_id: int
    best_model_name: str
    best_model_path: str
    trained_models: List[str]
    metrics: Dict[str, Any]

class PredictionRequest(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str

class PredictionResponse(BaseModel):
    dataset_id: int
    best_model_name: str
    prediction_label: str
    prediction_numeric: int
    probability_approved: Optional[float]
    processed_input: Dict[str, Any]

class BatchPredictionResponse(BaseModel):
    dataset_id: int
    best_model_name: str
    total_rows: int
    output_file: str
    preview: List[Dict[str, Any]]

class HistoryResponse(BaseModel):
    dataset_id: Optional[int] = None
    records: List[Dict[str, Any]]
