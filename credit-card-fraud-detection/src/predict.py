import pandas as pd
import joblib

from src.preprocessing import MODEL_DIR, transform_transactions


def predict_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return fraud labels and probabilities for rows in a transaction DataFrame."""
    model_path = MODEL_DIR / "fraud_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Model artifacts are missing. Run `python -m src.train` first.")

    model = joblib.load(model_path)
    prepared = transform_transactions(transactions)
    return pd.DataFrame(
        {
            "fraud_prediction": model.predict(prepared),
            "fraud_probability": model.predict_proba(prepared)[:, 1],
        },
        index=transactions.index,
    )