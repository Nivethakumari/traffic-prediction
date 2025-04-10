import streamlit as st
import pandas as pd
import joblib
import datetime
import pytz

# Load model and helpers
@st.cache_resource
def load_model():
    model = joblib.load("xgb_model.pkl")
    features = joblib.load("features_list.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, features, label_encoder

model, expected_columns, label_encoder = load_model()

# Junction name mapping
junction_map = {
    '1': 'Hebbal',
    '2': 'KR Puram',
    '3': 'Electronic City',
    '4': 'Nagawara'
}

# UI
st.set_page_config(page_title="Traffic Predictor", layout="centered")
st.title("🚦 Real-Time Traffic Prediction")
st.write("Enter your junction and see the predicted traffic level!")

# Real-time IST info
ist = pytz.timezone('Asia/Kolkata')
current_time = datetime.datetime.now(ist)
hour = current_time.hour
day = current_time.day
month = current_time.month
year = current_time.year
weekday = current_time.weekday()  # 0 = Monday, 6 = Sunday

# Feature engineering
is_weekend = 1 if weekday >= 5 else 0
is_month_start = 1 if day <= 3 else 0
is_month_end = 1 if day >= 28 else 0
quarter = (month - 1) // 3 + 1
is_weekend_morning = 1 if is_weekend and (5 <= hour <= 10) else 0

# Part of day
def get_part_of_day(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 20:
        return "Evening"
    else:
        return "Night"

part_of_day = get_part_of_day(hour)

# User input
junction_name = st.selectbox("Select a Junction", list(junction_map.values()))

# Build input dataframe
input_dict = {
    'Hour': hour,
    'DayOfWeek': weekday,
    'Month': month,
    'Year': year,
    'IsWeekend': is_weekend,
    'IsMonthStart': is_month_start,
    'IsMonthEnd': is_month_end,
    'Quarter': quarter,
    'IsWeekendMorning': is_weekend_morning,
    'PartOfDay_' + part_of_day: 1,
    'JunctionName_' + junction_name: 1
}

# Fill missing dummy columns with 0
for col in expected_columns:
    if col not in input_dict:
        input_dict[col] = 0

# Create DataFrame
input_data = pd.DataFrame([input_dict])[expected_columns]

# Predict
prediction = model.predict(input_data)
predicted_label = label_encoder.inverse_transform(prediction)[0]

# Display
st.markdown("---")
st.subheader(f"📍 Junction: {junction_name}")
st.subheader(f"🕒 Time: {current_time.strftime('%d %B %Y, %A – %I:%M %p')}")
st.success(f"**Predicted Traffic Level:** {predicted_label}")

# Footer
st.markdown("---")
st.markdown("<center><small>Created by Nivethakumari</small></center>", unsafe_allow_html=True)

