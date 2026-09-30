# Credit Card Fraud Detection

A small, end-to-end starter project for training and using a binary fraud classifier on the European credit-card transactions dataset.

## Project files

- `data/creditcard.csv`: input dataset, not included. It must contain a `Class` target column, where `1` means fraud, and the transaction features including `Time` and `Amount`.
- `notebooks/fraud_detection_eda.ipynb`: exploratory analysis.
- `src/`: preprocessing, training, evaluation, and batch prediction code.
- `models/`: generated model and scaler artifacts, created by training.
- `app.py`: Streamlit interface for uploading transactions and reviewing predictions.

## Setup

From this directory, create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place `creditcard.csv` in `data/`. Train and evaluate the model with:

```powershell
python -m src.train --data data/creditcard.csv
```

The command writes `fraud_model.pkl`, `time_scaler.pkl`, and `amount_scaler.pkl` to `models/`.

Launch the batch-prediction interface after training:

```powershell
streamlit run app.py
```

Upload a CSV containing the same feature columns as the training data. The `Class` column is optional and ignored during prediction.

## Notes

The dataset and trained artifacts are intentionally not fabricated or committed by this scaffold. The default model uses class weighting; evaluate precision and recall for the intended operating threshold before using predictions in a real payment workflow.