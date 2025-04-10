# 🚦 Traffic Level Prediction App

A Streamlit web application that predicts traffic levels (Low, Medium, High) at different traffic junctions in Bangalore using date and time features.

## 📊 About the Project

This app uses an XGBoost classification model trained on engineered features such as:
- Day of the week
- Time of day
- Month, quarter, and year
- Weekend and month start/end indicators
- Junction names

The model provides real-time traffic level predictions to help visualize and analyze congestion trends.

## 🔧 Features
- Real-time prediction based on current date and time
- Clean and minimalistic UI with automatic feature processing
- Trained with 92% accuracy using XGBoost and feature engineering

## 🚀 How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/Nivethakumari/traffic-prediction.git
   cd traffic-prediction
