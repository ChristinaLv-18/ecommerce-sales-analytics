import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# загрузка данных
df = pd.read_csv("data/ecommerce_sales.csv")

# target
y = df["Profit"]

# признаки
X = df[[
    "Region",
    "City",
    "Category",
    "Sub-Category",
    "Payment Mode",
    "Quantity",
    "Unit Price",
    "Discount"
]]

# категориальные и числовые
cat_features = [
    "Region",
    "City",
    "Category",
    "Sub-Category",
    "Payment Mode"
]

num_features = [
    "Quantity",
    "Unit Price",
    "Discount"
]

# preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ],
    remainder="passthrough"
)

# model
model = LinearRegression()

pipe = Pipeline([
    ("preprocessing", preprocessor),
    ("model", model)
])

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# train
pipe.fit(X_train, y_train)

# predict
y_pred = pipe.predict(X_test)

# metrics
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# save model
joblib.dump(pipe, "models/profit_model.pkl")