import streamlit as st

from src.predict import predict_transactions


st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")
st.title("Credit Card Fraud Detection")
st.write("Upload transaction features to score them with the trained classifier.")

uploaded_file = st.file_uploader("Transaction CSV", type="csv")
if uploaded_file is not None:
    try:
        import pandas as pd

        transactions = pd.read_csv(uploaded_file)
        results = predict_transactions(transactions)
        output = transactions.join(results)
        fraud_count = int(results["fraud_prediction"].sum())
        first_metric, second_metric = st.columns(2)
        first_metric.metric("Transactions", len(output))
        second_metric.metric("Flagged as fraud", fraud_count)
        st.dataframe(output, use_container_width=True)
        st.download_button(
            "Download predictions",
            output.to_csv(index=False).encode("utf-8"),
            file_name="fraud_predictions.csv",
            mime="text/csv",
        )
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))