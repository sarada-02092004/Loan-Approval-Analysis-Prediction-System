from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from datetime import datetime
from database import Base

class DatasetRecord(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    row_count = Column(Integer, default=0)
    column_count = Column(Integer, default=0)
    columns_json = Column(Text, nullable=False)
    file_path = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

class ProcessedDatasetRecord(Base):
    __tablename__ = "processed_datasets"

    id = Column(Integer, primary_key=True, index=True)
    source_dataset_id = Column(Integer, nullable=False)
    row_count = Column(Integer, default=0)
    column_count = Column(Integer, default=0)
    file_path = Column(String, nullable=False)
    notes = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelRunRecord(Base):
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, index=True)
    source_dataset_id = Column(Integer, nullable=False)
    best_model_name = Column(String, nullable=False)
    best_model_f1 = Column(Float, nullable=True)
    best_model_accuracy = Column(Float, nullable=True)
    model_path = Column(String, nullable=False)
    metrics_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class PredictionLog(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String, default="manual")
    prediction = Column(String, nullable=True)
    probability = Column(Float, nullable=True)
    payload_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
