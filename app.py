import streamlit as st
import pandas as pd
import joblib
import datetime

# Load the model
@st.cache_resource
def load_model():
    model = joblib.load("xgb_model.pkl")
    return model

# Prediction mapping
label_mapping = {0: 'High', 1: 'Low', 2: 'Medium'}

model = load_model()

# App title
st.title("🚦 Real-Time Traffic Prediction")

# Current time details
current_time = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
hour = current_time.hour
day = current_time.day
month = current_time.month
year = current_time.year
weekday = current_time.weekday()  # Monday = 0
is_weekend = 1 if weekday >= 5 else 0
is_month_start = 1 if day <= 5 else 0
is_month_end = 1 if day >= 25 else 0
quarter = (month - 1) // 3 + 1

# Part of day
if 5 <= hour < 12:
    part_of_day = "Morning"
elif 12 <= hour < 17:
    part_of_day = "Afternoon"
elif 17 <= hour < 21:
    part_of_day = "Evening"
else:
    part_of_day = "Night"

# Weekend morning
is_weekend_morning = 1 if is_weekend == 1 and part_of_day == "Morning" else 0

# Junction selection
junction = st.selectbox("Select Junction", ["Hebbal", "KR Puram", "Electronic City", "Nagawara"])

# Feature engineering
input_data = pd.DataFrame({
    "Junction": [junction],
    "Hour": [hour],
    "DayOfWeek": [weekday],
    "Month": [month],
    "Year": [year],
    "IsWeekend": [is_weekend],
    "IsMonthStart": [is_month_start],
    "IsMonthEnd": [is_month_end],
    "IsWeekendMorning": [is_weekend_morning],
    "Quarter": [quarter],
    "PartOfDay": [part_of_day]
})

# One-hot encode PartOfDay and Junction
input_data = pd.get_dummies(input_data)

# Add missing columns
expected_columns = [
    'Hour', 'DayOfWeek', 'Month', 'Year', 'IsWeekend',
    'IsMonthStart', 'IsMonthEnd', 'IsWeekendMorning', 'Quarter',
    'PartOfDay_Afternoon', 'PartOfDay_Evening', 'PartOfDay_Morning', 'PartOfDay_Night',
    'Junction_Electronic City', 'Junction_Hebbal', 'Junction_KR Puram', 'Junction_Nagawara'
]

for col in expected_columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Ensure correct column order
input_data = input_data[expected_columns].astype(float)

# Predict
prediction = model.predict(input_data)
traffic_level = label_mapping[prediction[0]]

# Display results
st.subheader("📊 Prediction Result")
st.success(f"Predicted Traffic Level: **{traffic_level}**")

# Footer
st.markdown("---")
st.markdown("Created by Nivethakumari")
