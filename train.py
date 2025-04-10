import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load your traffic dataset
df = pd.read_csv("traffic.csv")

# Encode the target variable
label_encoder = LabelEncoder()
df["Traffic"] = label_encoder.fit_transform(df["Traffic"])

# Features and target
X = df.drop("Traffic", axis=1)
y = df["Traffic"]

# Store feature names
feature_names = X.columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model and related files
joblib.dump(model, "xgb_model.pkl")
joblib.dump(feature_names, "features_list.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
