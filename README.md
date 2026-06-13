# E-Commerce Sales Analytics

Проект представляет собой веб-приложение для анализа продаж интернет-магазина и обучения ML-модели для прогнозирования прибыли.

# Стек технологий

* Python
* Streamlit
* FastAPI
* Pandas
* Plotly
* Scikit-learn
* Git / GitHub

# Streamlit Dashboard

* анализ продаж и прибыли;
* фильтрация по региону, городу, категории, подкатегории, способу оплаты и месяцу;
* графики продаж и прибыли во времени;
* анализ продаж по регионам, городам и категориям;
* анализ топовых товаров;
* расчет Average Item Value.

# ML Training

* выбор признаков для обучения модели;
* выбор размера train/test split;
* отправка запроса на обучение модели в FastAPI;
* обучение Linear Regression на backend;
* расчет метрик MAE и R²;
* график Actual Profit vs Predicted Profit;
* сохранение истории обучений в CSV-файл.

# Архитектура проекта

backend/
    main.py
    models/
        train_model.py

frontend/
    app.py
    pages/
        ml.py

data/
    ecommerce_sales.csv

requirements.txt
README.md


# Запуск проекта

# 1. Установка зависимостей

pip install -r requirements.txt

# 2. Запуск FastAPI

python -m uvicorn backend.main:app --reload

Документация API доступна по адресу http://127.0.0.1:8000/docs

# 3. Запуск Streamlit

streamlit run frontend/app.py

# 4. ML-модель

Целевая переменная - Profit
Используемая модель - LinearRegression

Метрики качества:
* MAE — средняя абсолютная ошибка;
* R² — доля объясненной дисперсии.

# 5. API endpoints

POST /train
Запускает обучение модели и возвращает метрики качества

GET /history
Возвращает историю запусков обучения
