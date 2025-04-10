import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Page config
st.set_page_config(page_title="Traffic Level Predictor", layout="centered")

# Load model and label encoder
@st.cache_resource
def load_model():
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("features_list.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

model, label_encoder = load_model()

# Junction Mapping
junction_map = {
    "Hebbal Junction": 1,
    "Nagawara Junction": 2,
    "Silk Board": 3,
    "Electronic City": 4
}

# Title
st.title("🚦 Real-Time Traffic Level Predictor")
st.markdown("Predict traffic levels for any date, time, and location in Bengaluru.")

# Inputs
junction_name = st.selectbox("Select Junction", list(junction_map.keys()))
junction = junction_map[junction_name]

selected_date = st.date_input("Select Date")
selected_hour = st.slider("Select Hour (0-23)", 0, 23, datetime.now().hour)

# Feature Engineering
day = selected_date.day
month = selected_date.month
weekday_name = selected_date.strftime("%A")
weekday_num = selected_date.weekday()
is_weekend = 1 if weekday_name in ["Saturday", "Sunday"] else 0
is_month_start = 1 if day <= 3 else 0
is_month_end = 1 if day >= 28 else 0
quarter = (month - 1) // 3 + 1
is_weekend_morning = 1 if is_weekend and (6 <= selected_hour <= 11) else 0

def get_part_of_day(hour):
    if 0 <= hour < 6:
        return 0  # Night
    elif 6 <= hour < 12:
        return 1  # Morning
    elif 12 <= hour < 18:
        return 2  # Afternoon
    else:
        return 3  # Evening

part_of_day = get_part_of_day(selected_hour)

# Display info
formatted_date = selected_date.strftime("%d %B %Y")
st.markdown(f"📅 **Selected Date:** {formatted_date} ({weekday_name})")
st.markdown(f"🕒 **Selected Hour:** {selected_hour}:00")

# Predict
if st.button("Predict Traffic Level"):
    input_data = pd.DataFrame([{
        "Junction": junction,
        "Hour": selected_hour,
        "Day": day,
        "Weekday": weekday_num,
        "Month": month,
        "IsWeekend": is_weekend,
        "PartOfDay": part_of_day,
        "IsMonthStart": is_month_start,
        "IsMonthEnd": is_month_end,
        "Quarter": quarter,
        "IsWeekendMorning": is_weekend_morning
    }])

    # Match expected columns
    expected_columns = model.get_booster().feature_names
    try:
        input_data = input_data[expected_columns].astype(float)
    except KeyError as e:
        st.error(f"❌ Feature mismatch error: {e}")
        st.write("📦 Input data columns:", input_data.columns.tolist())
        st.write("🧩 Model expects columns:", expected_columns)
        st.stop()

    # Predict
    prediction = model.predict(input_data)
    traffic_level = label_encoder.inverse_transform(prediction)[0]

    st.success(f"🚗 Predicted Traffic Level: **{traffic_level}**")

    # Custom messages
    if junction_name == "Hebbal Junction":
        st.info("🔍 Hebbal is known for high traffic even with fewer vehicles.")
    elif junction_name == "Nagawara Junction":
        st.info("🔍 Nagawara often shows medium traffic based on data patterns.")
    elif junction_name == "Electronic City":
        st.info("🔍 Electronic City is usually low traffic during non-peak hours.")

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
