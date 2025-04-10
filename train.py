import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import pickle

# Load dataset
df = pd.read_csv("traffic.csv")

# Feature Engineering
df["DayOfWeek"] = pd.to_datetime(df["Date"]).dt.weekday
df["Month"] = pd.to_datetime(df["Date"]).dt.month
df["Year"] = pd.to_datetime(df["Date"]).dt.year
df["Hour"] = pd.to_datetime(df["Time"]).dt.hour
df["IsWeekend"] = df["DayOfWeek"].apply(lambda x: 1 if x >= 5 else 0)
df["IsMonthStart"] = pd.to_datetime(df["Date"]).dt.is_month_start.astype(int)
df["IsMonthEnd"] = pd.to_datetime(df["Date"]).dt.is_month_end.astype(int)
df["Quarter"] = pd.to_datetime(df["Date"]).dt.quarter

# PartOfDay Feature
def get_part_of_day(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"

df["PartOfDay"] = df["Hour"].apply(get_part_of_day)

# Weekend Morning feature
df["IsWeekendMorning"] = ((df["IsWeekend"] == 1) & (df["PartOfDay"] == "Morning")).astype(int)

# Encoding Target
label_encoder = LabelEncoder()
df["TrafficLevel"] = label_encoder.fit_transform(df["TrafficLevel"])  # Assuming column name is TrafficLevel

# One-Hot Encoding
df = pd.get_dummies(df, columns=["PartOfDay", "Junction"], prefix=["PartOfDay", "JunctionName"])

# Features to use
features = [
    "Hour", "DayOfWeek", "Month", "Year", "IsWeekend",
    "IsMonthStart", "IsMonthEnd", "IsWeekendMorning", "Quarter"
] + [col for col in df.columns if col.startswith("PartOfDay_") or col.startswith("JunctionName_")]

X = df[features]
y = df["TrafficLevel"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# Save model
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save feature list
with open("features_list.pkl", "wb") as f:
    pickle.dump(features, f)

# Save label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Model, features, and label encoder saved successfully!")
