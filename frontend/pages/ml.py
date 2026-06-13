import streamlit as st
import pandas as pd
import requests
import plotly.express as px

#Настройка страницы Streamlit
st.set_page_config(page_title="Model Training", layout="wide")

st.title("Model Training")

#Адрес FastAPI сервиса
API_URL = "http://127.0.0.1:8000"

#Список доступных признаков для обучения модели
all_features = [
    "Region",
    "City",
    "Category",
    "Sub-Category",
    "Payment Mode",
    "Quantity",
    "Unit Price",
    "Discount"]

#Сохраняем выбранные признаки между обновлениями страницы
if "selected_features" not in st.session_state:
    st.session_state.selected_features = []

#Сохраняем выбранный размер обучающей выборки
if "train_size" not in st.session_state:
    st.session_state.train_size = 0.8

#Сохраняем результаты последнего обучения
if "result" not in st.session_state:
    st.session_state.result = None

#Выбор признаков для обучения модели
features = st.multiselect("Features for training",all_features, key="selected_features")

#Выбор доли данных для обучения
train_size = st.slider(
    "Train size",
    min_value=0.5,
    max_value=0.9,
    step=0.05,
    key="train_size")

#Запуск обучения модели
if st.button("Train Model"):

#Проверка, что пользователь выбрал признаки
    if not features:
        st.error("Select at least one feature for training.")
    else:
        # Формирование запроса для FastAPI
        payload = {
            "features": features,
            "train_size": train_size}
        # Отправка запроса на обучение модели
        with st.spinner("Training model on FastAPI..."):
            response = requests.post(f"{API_URL}/train",json=payload)
        #Сохранение результата обучения
        if response.status_code == 200:
            st.session_state.result = response.json()
            st.success("Training completed")
        else:
            st.error("Training failed")

#Отображение результатов обучения
if st.session_state.result is not None:

    result = st.session_state.result

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", result["mae"])
    col2.metric("R²", result["r2"])
    col3.metric("Features Used", result["features_used"])

    results_df = pd.DataFrame({
        "Actual Profit": result["actual"],
        "Predicted Profit": result["predicted"]})
    
#Сравнение реальных и предсказанных значений
    fig = px.scatter(
        results_df,
        x="Actual Profit",
        y="Predicted Profit",
        title="Actual vs Predicted Profit")

    st.plotly_chart(fig, use_container_width=True)


st.divider()
st.subheader("Training History")

# Загрузка истории запусков модели
if st.button("Show Training History"):

    history_response = requests.get(f"{API_URL}/history")

    if history_response.status_code == 200:
        history = history_response.json()

        if history:
            history_df = pd.DataFrame(history)
            st.dataframe(history_df, use_container_width=True)
        else:
            st.info("Training history is empty.")
    else:
        st.error("Could not load training history.")