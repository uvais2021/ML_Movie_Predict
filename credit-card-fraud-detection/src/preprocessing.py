from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
TARGET_COLUMN = "Class"


def prepare_data(data_path: str | Path, test_size: float = 0.2, random_state: int = 42):
    """Load, split, and scale the transaction data; fit scalers on training rows only."""
    data = pd.read_csv(data_path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must contain a '{TARGET_COLUMN}' target column.")
    if not {"Time", "Amount"}.issubset(data.columns):
        raise ValueError("Dataset must contain 'Time' and 'Amount' feature columns.")

    features = data.drop(columns=[TARGET_COLUMN])
    target = data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for column, filename in (("Time", "time_scaler.pkl"), ("Amount", "amount_scaler.pkl")):
        scaler = StandardScaler()
        x_train.loc[:, column] = scaler.fit_transform(x_train[[column]]).ravel()
        x_test.loc[:, column] = scaler.transform(x_test[[column]]).ravel()
        joblib.dump(scaler, MODEL_DIR / filename)

    return x_train, x_test, y_train, y_test


def transform_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Apply the saved feature scaling to transactions in training feature order."""
    model_path = MODEL_DIR / "fraud_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Model artifacts are missing. Train the model first.")
    model = joblib.load(model_path)
    feature_columns = list(model.feature_names_in_)

    missing = sorted(set(feature_columns) - set(transactions.columns))
    if missing:
        raise ValueError(f"Input is missing required feature columns: {', '.join(missing)}")

    prepared = transactions.loc[:, feature_columns].copy()
    for column, filename in (("Time", "time_scaler.pkl"), ("Amount", "amount_scaler.pkl")):
        scaler_path = MODEL_DIR / filename
        if not scaler_path.exists():
            raise FileNotFoundError(f"Missing scaler artifact: {scaler_path}")
        scaler = joblib.load(scaler_path)
        prepared.loc[:, column] = scaler.transform(prepared[[column]]).ravel()
    return prepared