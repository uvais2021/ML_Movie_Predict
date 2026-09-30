import argparse
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression

from src.evaluate import evaluate_model
from src.preprocessing import MODEL_DIR, prepare_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a credit-card fraud classifier.")
    parser.add_argument("--data", type=Path, default=Path("data/creditcard.csv"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    x_train, x_test, y_train, y_test = prepare_data(
        args.data, test_size=args.test_size, random_state=args.random_state
    )
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=args.random_state)
    model.fit(x_train, y_train)
    evaluate_model(model, x_test, y_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "fraud_model.pkl")
    print(f"Saved model artifacts to {MODEL_DIR}")


if __name__ == "__main__":
    main()