import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv("traffic.csv")

# Encode target labels
label_encoder = LabelEncoder()
df['TrafficLevel'] = label_encoder.fit_transform(df['TrafficLevel'])

# One-hot encode categorical features
X = pd.get_dummies(df.drop("TrafficLevel", axis=1))
y = df["TrafficLevel"]

# Save feature names
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, "features_list.pkl")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier()
model.fit(X_train, y_train)

# Save model and label encoder
joblib.dump(model, "xgb_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
