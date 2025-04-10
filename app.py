import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import pytz

# Load model, features, and label encoder
@st.cache_resource
def load_model():
    model = joblib.load("xgb_model.pkl")
    features = joblib.load("features_list.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, features, label_encoder

model, feature_list, label_encoder = load_model()

# Title and description
st.title("🚦 Real-Time Traffic Prediction")
st.write("This app predicts traffic levels (Low, Medium, High) based on the current time and location.")

# Get current IST date and time
ist = pytz.timezone("Asia/Kolkata")
now = datetime.now(ist)
day = now.strftime("%A")
date = now.strftime("%d %B %Y")
hour = now.hour
minute = now.minute

# Display current date and time
st.subheader("📅 Date and Time Info")
st.write(f"**Date:** {date}")
st.write(f"**Day:** {day}")
st.write(f"**Time:** {hour:02d}:{minute:02d} IST")

# Map junction names
junction_map = {
    "J1": "Hebbal Junction",
    "J2": "KR Puram Junction",
    "J3": "Marathahalli Junction",
    "J4": "Silk Board Junction"
}

junction_code = st.selectbox("🛣️ Select Junction", list(junction_map.keys()), format_func=lambda x: junction_map[x])

# Feature Engineering
def extract_features(current_time, junction_code):
    hour = current_time.hour
    minute = current_time.minute
    part_of_day = (
        "Night" if 0 <= hour < 6 else
        "Morning" if 6 <= hour < 12 else
        "Afternoon" if 12 <= hour < 17 else
        "Evening"
    )
    
    is_month_start = int(current_time.day <= 3)
    is_month_end = int(current_time.day >= 28)
    is_weekend_morning = int(current_time.weekday() >= 5 and 6 <= hour <= 10)
    quarter = (current_time.month - 1) // 3 + 1

    features = {
        "Junction": junction_code,
        "Day": current_time.strftime("%A"),
        "Hour": hour,
        "Minute": minute,
        "PartOfDay": part_of_day,
        "IsMonthStart": is_month_start,
        "IsMonthEnd": is_month_end,
        "IsWeekendMorning": is_weekend_morning,
        "Quarter": quarter,
        "Month": current_time.month,
        "Year": current_time.year
    }
    return features

# Prepare input data
features = extract_features(now, junction_code)
input_df = pd.DataFrame([features])

# One-hot encode categorical features
input_data = pd.get_dummies(input_df)

# Align input with training features
for col in feature_list:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[feature_list]

# Predict button
if st.button("Predict Traffic"):
    prediction = model.predict(input_data)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    st.success(f"🚗 **Predicted Traffic Level:** {predicted_label}")
    
# Footer
st.markdown("---")
st.markdown("Created by **Nivethakumari**")
