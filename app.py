import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Traffic Prediction", layout="wide")
st.title("🚦 Real-Time Traffic Level Predictor")

@st.cache_resource
def load_model():
    model = joblib.load("xgb_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, label_encoder

model, label_encoder = load_model()

@st.cache_data
def load_data():
    return pd.read_csv("traffic.csv")

data = load_data()

# ---------- VISUALIZATION FUNCTIONS ----------
def plot_traffic_heatmap(df):
    heatmap_data = df.groupby(["Hour", "Junction"]).size().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, cmap="YlGnBu", annot=False, fmt=".0f", ax=ax)
    ax.set_title("Heatmap of Traffic Volume by Hour and Junction")
    st.pyplot(fig)

def plot_hourly_trend(df):
    hourly_trend = df.groupby("Hour").size()
    fig, ax = plt.subplots()
    hourly_trend.plot(kind="line", marker='o', ax=ax)
    ax.set_title("Hourly Traffic Volume Trend")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Traffic Volume")
    st.pyplot(fig)

def plot_junction_traffic(df):
    junction_avg = df.groupby("Junction").size()
    fig, ax = plt.subplots()
    junction_avg.plot(kind="bar", color="orange", ax=ax)
    ax.set_title("Average Traffic by Junction")
    st.pyplot(fig)

def plot_traffic_distribution(df):
    fig, ax = plt.subplots()
    df["TrafficLevel"].value_counts().plot(kind="pie", autopct='%1.1f%%', ax=ax, startangle=90)
    ax.set_ylabel("")
    ax.set_title("Traffic Level Distribution")
    st.pyplot(fig)

# ---------- UI ELEMENTS ----------
st.markdown("## 📅 Current Date & Time")
now = datetime.datetime.now()
st.write("**Date:**", now.strftime("%d %B %Y"))
st.write("**Day:**", now.strftime("%A"))
st.write("**Time:**", now.strftime("%H:%M:%S"))

st.markdown("## 🧮 Predict Traffic Level")
junction = st.selectbox("Select Junction", ["Hebbal", "KR Puram", "Electronic City", "Nagawara"])

hour = now.hour
day = now.day
month = now.month
year = now.year
weekday = now.weekday()
part_of_day = (
    "Morning" if 5 <= hour < 12 else
    "Afternoon" if 12 <= hour < 17 else
    "Evening" if 17 <= hour < 21 else
    "Night"
)

is_month_start = int(now.day <= 5)
is_month_end = int(now.day >= 25)
is_weekend = int(now.weekday() >= 5)
is_weekend_morning = int(is_weekend and hour < 12)
quarter = (now.month - 1) // 3 + 1

input_dict = {
    'Junction': junction,
    'Hour': hour,
    'DayOfWeek': weekday,
    'Month': month,
    'Year': year,
    'IsWeekend': is_weekend,
    'IsMonthStart': is_month_start,
    'IsMonthEnd': is_month_end,
    'IsWeekendMorning': is_weekend_morning,
    'Quarter': quarter,
    'PartOfDay': part_of_day
}

input_df = pd.DataFrame([input_dict])

# One-hot encoding for categorical features
input_encoded = pd.get_dummies(input_df, columns=["PartOfDay", "Junction"], prefix=["PartOfDay", "JunctionName"])

# Align to expected model input
expected_columns = [
    "Junction", "Hour", "DayOfWeek", "Month", "Year", "IsWeekend", "IsMonthStart",
    "IsMonthEnd", "IsWeekendMorning", "Quarter",
    "PartOfDay_Afternoon", "PartOfDay_Evening", "PartOfDay_Morning", "PartOfDay_Night",
    "JunctionName_Electronic City", "JunctionName_Hebbal",
    "JunctionName_KR Puram", "JunctionName_Nagawara"
]

# Handle missing columns
for col in expected_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded.reindex(columns=expected_columns, fill_value=0)

# Prediction
prediction = model.predict(input_encoded)
traffic_level = label_encoder.inverse_transform(prediction)[0]
st.success(f"Predicted Traffic Level at {junction}: **{traffic_level}**")

# ---------- VISUALIZATION SECTION ----------
st.markdown("## 📊 Data Visualizations")
if st.checkbox("Show Data Visualizations"):
    st.markdown("### 🔥 Heatmap: Hour vs Junction")
    plot_traffic_heatmap(data)

    st.markdown("### ⏱️ Line Chart: Hourly Traffic Trend")
    plot_hourly_trend(data)

    st.markdown("### 🚦 Bar Chart: Avg Traffic by Junction")
    plot_junction_traffic(data)

    st.markdown("### 🧩 Pie Chart: Traffic Level Distribution")
    plot_traffic_distribution(data)

# Footer
st.markdown("---")
st.markdown("Created by **Nivethakumari and Dharshini Shree** ✨")

