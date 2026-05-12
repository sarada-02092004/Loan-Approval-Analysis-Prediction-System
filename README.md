# Loan Approval Intelligence Platform — Phase 1 + Phase 2 + Phase 3 + Phase 4

This version extends the project with operational features on top of the existing modeling system.

## Included
- FastAPI backend
- SQLite persistence
- CSV upload and schema validation
- Dataset summary and missing-value analysis
- Preprocessing with feature engineering
- Full EDA dashboard
- Model training and comparison
- Best model persistence with joblib
- Single-loan approval prediction
- Batch prediction with CSV export
- Model run history
- Prediction history
- Download endpoints for processed dataset, trained model, and batch outputs
- Premium Streamlit frontend

## Models
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

## Phase 4 features
- Batch prediction endpoint and UI
- CSV output export for batch scoring
- Downloadable processed dataset
- Downloadable best model artifact
- Model run history from SQLite
- Prediction history from SQLite

## Run backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r ..\requirements.txt
uvicorn main:app --reload
```

## Run frontend
```bash
cd frontend
..\backend\venv\Scripts\activate
streamlit run app.py
```

## Endpoints
- `POST /upload-data`
- `GET /summary/{dataset_id}`
- `GET /missing-values/{dataset_id}`
- `POST /preprocess/{dataset_id}`
- `GET /eda/basic/{dataset_id}`
- `GET /eda/univariate/{dataset_id}`
- `GET /eda/bivariate/{dataset_id}`
- `GET /eda/multivariate/{dataset_id}`
- `GET /eda/class-imbalance/{dataset_id}`
- `GET /eda/outliers/{dataset_id}`
- `GET /eda/correlation/{dataset_id}`
- `GET /eda/findings/{dataset_id}`
- `POST /train-models/{dataset_id}`
- `POST /predict/{dataset_id}`
- `POST /predict-batch/{dataset_id}`
- `GET /history/models/{dataset_id}`
- `GET /history/predictions/{dataset_id}`
- `GET /download/model/{dataset_id}`
- `GET /download/batch/{dataset_id}`
- `GET /download/processed/{dataset_id}`

## Typical flow
1. Upload dataset
2. Review summary and missing values
3. Run preprocessing
4. Explore EDA dashboard
5. Train and compare models
6. Predict single loan approval
7. Score batch files and download results
8. Review model and prediction history
