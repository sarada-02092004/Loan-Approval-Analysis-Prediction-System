from pathlib import Path
from uuid import uuid4
import pandas as pd

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def save_uploaded_file(upload_file) -> Path:
    suffix = Path(upload_file.filename).suffix.lower() or ".csv"
    path = DATA_DIR / f"{uuid4().hex}{suffix}"
    with open(path, "wb") as f:
        f.write(upload_file.file.read())
    return path

def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def safe_mode(series):
    mode = series.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else None
