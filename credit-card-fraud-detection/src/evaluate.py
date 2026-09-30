from sklearn.metrics import classification_report, roc_auc_score


def evaluate_model(model, x_test, y_test) -> dict[str, float]:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    metrics = {
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"],
        "roc_auc": roc_auc_score(y_test, probabilities),
    }
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    return metrics