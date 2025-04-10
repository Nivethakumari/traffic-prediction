import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Page config
st.set_page_config(page_title="Traffic Level Predictor", layout="centered")

@st.cache_resource
def load_model():
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("features_list.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

model, label_encoder = load_model()

# Model expected columns
expected_columns = [
    "Junction", "Hour", "DayOfWeek", "Month", "Year", "IsWeekend",
    "IsMonthStart", "IsMonthEnd", "IsWeekendMorning", "Quarter",
    "PartOfDay_Afternoon", "PartOfDay_Evening", "PartOfDay_Morning", "PartOfDay_Night",
    "JunctionName_Electronic City", "JunctionName_Hebbal", "JunctionName_KR Puram", "JunctionName_Nagawara"
]

junction_options = ["Hebbal", "Nagawara", "KR Puram", "Electronic City"]

# Title
st.title("🚦 Real-Time Traffic Level Predictor")
st.markdown("Predict traffic levels for any date, time, and location in Bengaluru.")

# Inputs
junction_name = st.selectbox("Select Junction", junction_options)
selected_date = st.date_input("Select Date")
selected_hour = st.slider("Select Hour (0-23)", 0, 23, datetime.now().hour)

# Feature Engineering
day = selected_date.day
month = selected_date.month
year = selected_date.year
day_of_week = selected_date.weekday()
is_weekend = 1 if selected_date.strftime("%A") in ["Saturday", "Sunday"] else 0
is_month_start = 1 if day <= 3 else 0
is_month_end = 1 if day >= 28 else 0
quarter = (month - 1) // 3 + 1
is_weekend_morning = 1 if is_weekend and (6 <= selected_hour <= 11) else 0

# Part of Day One-Hot Encoding
part_of_day = ""
if 0 <= selected_hour < 6:
    part_of_day = "Night"
elif 6 <= selected_hour < 12:
    part_of_day = "Morning"
elif 12 <= selected_hour < 18:
    part_of_day = "Afternoon"
else:
    part_of_day = "Evening"

part_of_day_encoded = {
    "PartOfDay_Morning": 1 if part_of_day == "Morning" else 0,
    "PartOfDay_Afternoon": 1 if part_of_day == "Afternoon" else 0,
    "PartOfDay_Evening": 1 if part_of_day == "Evening" else 0,
    "PartOfDay_Night": 1 if part_of_day == "Night" else 0
}

# Junction Name One-Hot Encoding
junction_encoded = {
    "JunctionName_Hebbal": 1 if junction_name == "Hebbal" else 0,
    "JunctionName_Nagawara": 1 if junction_name == "Nagawara" else 0,
    "JunctionName_KR Puram": 1 if junction_name == "KR Puram" else 0,
    "JunctionName_Electronic City": 1 if junction_name == "Electronic City" else 0
}

# Combine all features into a single dictionary
input_dict = {
    "Junction": junction_options.index(junction_name) + 1,
    "Hour": selected_hour,
    "DayOfWeek": day_of_week,
    "Month": month,
    "Year": year,
    "IsWeekend": is_weekend,
    "IsMonthStart": is_month_start,
    "IsMonthEnd": is_month_end,
    "IsWeekendMorning": is_weekend_morning,
    "Quarter": quarter,
    **part_of_day_encoded,
    **junction_encoded
}

# Reorder to expected columns
input_data = pd.DataFrame([input_dict])[expected_columns]

# Display date and hour
formatted_date = selected_date.strftime("%d %B %Y")
st.markdown(f"📅 **Selected Date:** {formatted_date} ({selected_date.strftime('%A')})")
st.markdown(f"🕒 **Selected Hour:** {selected_hour}:00")

# Predict
if st.button("Predict Traffic Level"):
    prediction = model.predict(input_data)
    traffic_level = label_encoder.inverse_transform(prediction)[0]
    st.success(f"🚗 Predicted Traffic Level: **{traffic_level}**")

    # Visualization
    st.subheader("📊 Traffic Pattern Visualization")
    fig, ax = plt.subplots()
    sns.barplot(x=["Night", "Morning", "Afternoon", "Evening"], y=[10, 40, 70, 50], palette="viridis", ax=ax)
    ax.set_ylabel("Avg. Traffic Intensity (%)")
    ax.set_title("Sample Traffic Distribution by Time of Day")
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("👩‍💻 Created by **Nivethakumari & Dharshini Shree**")
