import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Page config
st.set_page_config(page_title="Traffic Level Predictor", layout="centered")

# Load model and expected feature list
@st.cache_resource
def load_model():
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("features_list.pkl", "rb") as f:
        feature_list = pickle.load(f)
    return model, feature_list

model, expected_columns = load_model()

# Junction mapping for display
junction_map = {
    "Hebbal": "JunctionName_Hebbal",
    "Nagawara": "JunctionName_Nagawara",
    "Electronic City": "JunctionName_Electronic City",
    "KR Puram": "JunctionName_KR Puram"
}

# Title
st.title("🚦 Real-Time Traffic Level Predictor")
st.markdown("Predict traffic levels for any date, time, and location in Bengaluru.")

# User input
junction_name = st.selectbox("Select Junction", list(junction_map.keys()))
selected_date = st.date_input("Select Date")
selected_hour = st.slider("Select Hour (0-23)", 0, 23, datetime.now().hour)

# Feature engineering
day = selected_date.day
month = selected_date.month
year = selected_date.year
weekday_num = selected_date.weekday()
is_weekend = 1 if weekday_num >= 5 else 0
is_month_start = 1 if day <= 3 else 0
is_month_end = 1 if day >= 28 else 0
quarter = (month - 1) // 3 + 1
is_weekend_morning = 1 if is_weekend and (6 <= selected_hour <= 11) else 0

# Part of Day One-Hot Encoding
part_of_day = {
    "PartOfDay_Night": 0,
    "PartOfDay_Morning": 0,
    "PartOfDay_Afternoon": 0,
    "PartOfDay_Evening": 0
}
if 0 <= selected_hour < 6:
    part_of_day["PartOfDay_Night"] = 1
elif 6 <= selected_hour < 12:
    part_of_day["PartOfDay_Morning"] = 1
elif 12 <= selected_hour < 18:
    part_of_day["PartOfDay_Afternoon"] = 1
else:
    part_of_day["PartOfDay_Evening"] = 1

# Junction One-Hot Encoding
junction_encoding = {
    "JunctionName_Hebbal": 0,
    "JunctionName_Nagawara": 0,
    "JunctionName_Electronic City": 0,
    "JunctionName_KR Puram": 0
}
junction_encoding[junction_map[junction_name]] = 1

# Build input data dictionary
input_dict = {
    "Junction": list(junction_map.keys()).index(junction_name) + 1,
    "Hour": selected_hour,
    "DayOfWeek": weekday_num,
    "Month": month,
    "Year": year,
    "IsWeekend": is_weekend,
    "IsMonthStart": is_month_start,
    "IsMonthEnd": is_month_end,
    "IsWeekendMorning": is_weekend_morning,
    "Quarter": quarter
}
input_dict.update(part_of_day)
input_dict.update(junction_encoding)

# Convert to DataFrame
input_data = pd.DataFrame([input_dict])
input_data = input_data.reindex(columns=expected_columns, fill_value=0).astype(float)

# Display selected info
formatted_date = selected_date.strftime("%d %B %Y (%A)")
st.markdown(f"📅 **Selected Date:** {formatted_date}")
st.markdown(f"🕒 **Selected Hour:** {selected_hour}:00")
st.markdown(f"📍 **Junction:** {junction_name}")

# Debug info if needed
if st.checkbox("Show model input data"):
    st.subheader("🧪 Model Input Features")
    st.dataframe(input_data)

# Predict
if st.button("Predict Traffic Level"):
    prediction = model.predict(input_data)
    label_map = {0: "Low", 1: "Medium", 2: "High"}
    traffic_level = label_map[int(prediction[0])]

    st.success(f"🚗 Predicted Traffic Level: **{traffic_level}**")

    # Notes
    if junction_name == "Hebbal":
        st.info("🔍 Hebbal usually has high traffic. But sometimes it may have Low traffic as well.")
    elif junction_name == "Nagawara":
        st.info("🔍 Nagawara tends to show 'Medium' traffic most of the time.")
    elif junction_name == "Electronic City":
        st.info("🔍 Electronic City often shows 'Medium' traffic levels.")
    elif junction_name == "KR Puram":
        st.info("🔍 KR Puram typically records 'Medium' traffic under most conditions.")

# Footer
st.markdown("---")
st.markdown("👩‍💻 Created by **Nivethakumari & Dharshini Shree**")

