from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

#Создание экземпляра FastAPI приложения
app = FastAPI()

#Создание экземпляра FastAPI приложения
class TrainRequest(BaseModel):
    features: list[str]
    train_size: float


@app.post("/train")
def train_model(request: TrainRequest):

    #Загрузка датасета
    df = pd.read_csv("data/ecommerce_sales.csv")

    #Формирование матрицы признаков
    X = df[request.features]
    y = df["Profit"]

    #Список категориальных признаков
    categorical_columns = [
        "Region",
        "City",
        "Category",
        "Sub-Category",
        "Payment Mode"]

    #Определяем какие категориальные признаки выбрал пользователь
    cat_features = [
        col for col in request.features
        if col in categorical_columns]

    # Предобработка данных
    # One-Hot Encoding для категориальных признаков
    preprocessor = ColumnTransformer(transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)],remainder="passthrough")

    model = Pipeline([("preprocessor", preprocessor),("model", LinearRegression())])

    #Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=request.train_size,random_state=42)

    #Обучение модели
    model.fit(X_train, y_train)

    #Получение прогнозов
    y_pred = model.predict(X_test)

    #Расчет метрик качества модели
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    history_row = pd.DataFrame([{
        "train_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": "LinearRegression",
        "dataset_name": "ecommerce_sales.csv",
        "train_size": request.train_size,
        "features": ", ".join(request.features),
        "features_count": len(request.features),
        "mae": round(mae, 2),
        "r2": round(r2, 3)}])

    #Путь к файлу с историей запусков модели
    history_path = "backend/training_history.csv"

    #Проверка существования файла
    file_exists = os.path.exists(history_path)
    file_is_empty = file_exists and os.path.getsize(history_path) == 0

    #Сохранение информации о запуске модели
    history_row.to_csv(history_path,mode="a",header=not os.path.exists(history_path),index=False)

    #Возврат результатов на frontend
    return {
        "mae": round(mae, 2),
        "r2": round(r2, 3),
        "features_used": len(request.features),
        "actual": y_test.tolist(),
        "predicted": y_pred.tolist()}


@app.get("/history")
def get_history():

    history_path = "backend/training_history.csv"

    if not os.path.exists(history_path):
        return []

    history = pd.read_csv(history_path)

    return history.to_dict(orient="records")