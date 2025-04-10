import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Traffic Level Predictor", layout="centered")

# Load model and label encoder
@st.cache_resource

def load_model():
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

model, le = load_model()

# Junction Mapping
junction_map = {
    "Hebbal Junction": 1,
    "Nagawara Junction": 2,
    "Silk Board": 3,
    "Electronic City": 4
}

# Title and description
st.title("🚦 Real-Time Traffic Level Predictor")
st.markdown("Predict traffic levels for any date, time, and location in Bengaluru.")

# User Inputs
junction_name = st.selectbox("Select Junction", list(junction_map.keys()))
junction = junction_map[junction_name]

selected_date = st.date_input("Select Date")
selected_hour = st.slider("Select Hour (0-23)", 0, 23, datetime.now().hour)

# Feature engineering
day = selected_date.day
month = selected_date.month
weekday_name = selected_date.strftime("%A")
weekday_num = selected_date.weekday()
is_weekend = 1 if weekday_name in ["Saturday", "Sunday"] else 0
is_month_start = 1 if day <= 3 else 0
is_month_end = 1 if day >= 28 else 0
quarter = (month - 1) // 3 + 1
is_weekend_morning = 1 if is_weekend and (6 <= selected_hour <= 11) else 0

# Part of Day
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

# Display selected info
formatted_date = selected_date.strftime("%d %B %Y")
st.markdown(f"📅 **Selected Date:** {formatted_date} ({weekday_name})")
st.markdown(f"🕒 **Selected Hour:** {selected_hour}:00")

# Debug checkbox
debug = st.checkbox("Show model input data")

# Predict button
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

    # Ensure columns match model
    expected_columns = model.get_booster().feature_names
    input_data = input_data[expected_columns].astype(float)

    if debug:
        st.subheader("🧪 Model Input Data")
        st.write(input_data)
        st.write("🧩 Expected Features:", expected_columns)

    prediction = model.predict(input_data)
    traffic_level = le.inverse_transform(prediction)[0]

    st.success(f"🚗 Predicted Traffic Level: **{traffic_level}**")

    # Junction-specific interpretation
    if junction_name == "Hebbal Junction":
        st.info("🔍 Note: Predictions for **Hebbal** are made relative to its usual traffic levels. Even low vehicle counts may result in 'High' labels due to local patterns.")
    elif junction_name == "Nagawara Junction":
        st.info("🔍 Note: **Nagawara** tends to show 'Medium' traffic most of the time, based on learned data patterns.")
    elif junction_name == "Electronic City":
        st.info("🔍 Note: **Electronic City** is relatively less congested and often predicted as 'Low' traffic.")

    # Visualization
    st.subheader("📈 Traffic Probability Chart")
    probs = model.predict_proba(input_data)[0]
    labels = le.inverse_transform([0, 1, 2])  # Assuming order matches ['High', 'Low', 'Medium']
    prob_df = pd.DataFrame({"Traffic Level": labels, "Probability": probs})

    fig, ax = plt.subplots()
    sns.barplot(data=prob_df, x="Traffic Level", y="Probability", palette="coolwarm", ax=ax)
    ax.set_title("Prediction Confidence")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("👩‍💻 Created by **Nivethakumari & Dharshini Shree**")
