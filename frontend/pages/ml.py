import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(page_title="Model Training", layout="wide")

st.title("Model Training")

API_URL = "http://127.0.0.1:8000"

all_features = [
    "Region",
    "City",
    "Category",
    "Sub-Category",
    "Payment Mode",
    "Quantity",
    "Unit Price",
    "Discount"
]
if "selected_features" not in st.session_state:
    st.session_state.selected_features = []

if "train_size" not in st.session_state:
    st.session_state.train_size = 0.8

if "job_id" not in st.session_state:
    st.session_state.job_id = None

if "result" not in st.session_state:
    st.session_state.result = None


features = st.multiselect("Features for training",all_features,key="selected_features")

train_size = st.slider("Train size",0.5,0.9,step=0.05,key="train_size")

if st.button("Train Model"):

    if not features:
        st.error("Select at least one feature for training.")
    else:
        payload = {
            "features": features,
            "train_size": train_size}

        response = requests.post(
            f"{API_URL}/train",
            json=payload)

        data = response.json()

        st.session_state.job_id = data["job_id"]
        st.session_state.result = None

        st.success("Training started")


if st.session_state.job_id is not None:

    status_response = requests.get(
        f"{API_URL}/train/status/{st.session_state.job_id}")

    status_data = status_response.json()
    status = status_data["status"]
    st.info(f"Training status: {status}")

    if status == "completed":
        st.session_state.result = status_data["result"]

    elif status == "failed":
        st.error(status_data["error"])

if st.session_state.result is not None:

    result = st.session_state.result

    st.success("Training completed")

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", result["mae"])
    col2.metric("R²", result["r2"])
    col3.metric("Features Used", result["features_used"])

    results_df = pd.DataFrame({
        "Actual Profit": result["actual"],
        "Predicted Profit": result["predicted"]})

    fig = px.scatter(
        results_df,
        x="Actual Profit",
        y="Predicted Profit",
        title="Actual vs Predicted Profit")

    st.plotly_chart(fig, use_container_width=True)