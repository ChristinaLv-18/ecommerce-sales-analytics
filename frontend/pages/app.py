import streamlit as st
import pandas as pd
import plotly.express as px

#Даем название странице и настраиваем широкий режим отображения
st.set_page_config(page_title="E-Commerce Dashboard", layout="wide")

# CACHE (чтобы не грузилось заново)
@st.cache_data

#Преобразуем дату заказа в формат datetime
def load_data():
    df = pd.read_csv("data/ecommerce_sales.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
#Создаём колонку "Month" (год-месяц) для агрегаций
    df["Month"] = df["Order Date"].dt.to_period("M").astype(str)
    return df

#Загружаем данные из датасета
df = load_data()

#Создадим список цветов для графиков
custom_colors = [
    "#5B8FF9", 
    "#61DDAA",
    "#65789B",
    "#F6BD16",
    "#7262FD", 
    "#78D3F8"]

#Создадим словарь цветов по категориям
category_colors = {
    "Electronics": "#5B8FF9",
    "Furniture": "#61DDAA",
    "Clothing": "#F6BD16",
    "Office Supplies": "#7262FD",
    "Sports": "#78D3F8"}

#Настройка вариантов выбора в фильтрах
st.sidebar.title("Filters")

region = st.sidebar.multiselect("Region", sorted(df["Region"].unique()),key="region")

#Создадим динамический список городов (зависимость от выбранного региона)
if region:
    available_cities = sorted(df[df["Region"].isin(region)]["City"].unique())
else:
    available_cities = sorted(df["City"].unique())

city = st.sidebar.multiselect("City", available_cities, key="city")

category = st.sidebar.multiselect("Category", sorted(df["Category"].unique()),key="category")

sub_category = st.sidebar.multiselect("Sub-Category", sorted(df["Sub-Category"].unique()),key="sub_category")

payment_mode = st.sidebar.multiselect("Payment Mode", sorted(df["Payment Mode"].unique()),key="payment_mode")

selected_months = st.sidebar.multiselect("Months", sorted(df["Month"].unique()),key="months")

#Настроим фильтры в копии основного датасета с возможностью выбора нескольких вариантов
filtered_df = df.copy()

if region:
     filtered_df = filtered_df[filtered_df["Region"].isin(region)]
                               
if city:
    filtered_df = filtered_df[filtered_df["City"].isin(city)]

if category:
    filtered_df = filtered_df[filtered_df["Category"].isin(category)]

if sub_category:
    filtered_df = filtered_df[filtered_df["Sub-Category"].isin(sub_category)]

if payment_mode:
    filtered_df = filtered_df[filtered_df["Payment Mode"].isin(payment_mode)]

if selected_months:
    filtered_df = filtered_df[filtered_df["Month"].isin(selected_months)]

#Добавим сброс фильтров и кнопку "Reset filters"
def clear_filters():
    st.session_state.region = []
    st.session_state.city = []
    st.session_state.category = []
    st.session_state.sub_category = []
    st.session_state.payment_mode = []
    st.session_state.months = []

st.sidebar.button("Reset filters", on_click=clear_filters)

#Создадим заголовок дашборда 
st.title("📊 E-Commerce Dashboard")
#Добавим метрики Total Sales и Total Profit в начале страницы дэшборда
col1, col2 = st.columns(2)

col1.metric("Total Sales", f"{filtered_df['Sales'].sum():,.0f}")
col2.metric("Total Profit", f"{filtered_df['Profit'].sum():,.0f}")

#Создадим график Sales Over Time
sales_time = filtered_df.groupby("Month")["Sales"].sum().reset_index()

fig1 = px.line(
    sales_time,
    x="Month",
    y="Sales",
    markers=True,
    title="Sales Over Time")

avg_sales = sales_time["Sales"].mean()

fig1.add_hline(
    y=avg_sales,
    line_dash="dash",
    annotation_text="Average Sales")

with col1:
    st.plotly_chart(fig1, use_container_width=True)

#Создадим график Profit Over Time
profit_time = filtered_df.groupby("Month")["Profit"].sum().reset_index()

fig2 = px.line(
    profit_time,
    x="Month",
    y="Profit",
    markers=True,
    title="Profit Over Time")

avg_profit = profit_time["Profit"].mean()

fig2.add_hline(
    y=avg_profit,
    line_dash="dash",
    annotation_text="Average Profit")

fig2.update_traces(
    line=dict(width=3, color="#61DDAA"))

with col2:
    st.plotly_chart(fig2, use_container_width=True)

#Создадим график Sales by Region and City
col3, col4 = st.columns(2)

region_city_sales = (filtered_df.groupby(["Region", "City"])["Sales"].sum().reset_index())

fig3 = px.bar(
    region_city_sales,
    x="Region",
    y="Sales",
    color="City",
    color_discrete_sequence=custom_colors,
    title="Sales by Region and City")

with col3:
    st.plotly_chart(fig3, use_container_width=True)

#Создадим график Sales by Category
cat_sales = filtered_df.groupby("Category")["Sales"].sum().reset_index()

fig4 = px.bar(
    cat_sales,
    x="Category",
    y="Sales",
    color="Category",
    color_discrete_map=category_colors,
    title="Sales by Category")

with col4:
    st.plotly_chart(fig4, use_container_width=True)

#Создадим график Top 10 Products
col5, col6 = st.columns(2)
top_products = (filtered_df.groupby("Product Name")["Sales"].sum().reset_index().sort_values("Sales", ascending=False).head(10))

fig5 = px.bar(
    top_products,
    x="Product Name",
    y="Sales",
    color_discrete_sequence=custom_colors,
    title="Top 10 Products")

with col5:
    st.plotly_chart(fig5, use_container_width=True)

#Создадим график Average Item Value by Category
aiv = (filtered_df.groupby("Category")[["Sales", "Quantity"]].sum().reset_index())
aiv["Average Item Value"] = aiv["Sales"] / aiv["Quantity"]
    
fig6 = px.bar(
aiv,
x="Category",
y="Average Item Value",
color="Category",
color_discrete_map=category_colors,
title="Average Item Value by Category")

with col6:
    st.plotly_chart(fig6, use_container_width=True)