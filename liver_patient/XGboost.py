import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np

# 1. Data Initialization (Based on the provided snippet)

df = pd.read_csv('liver_patient.csv')

# --- 2. Data Preprocessing ---

# A. Handle Categorical Feature (Gender)
# Encode Gender: Male=1, Female=0
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# B. Separate Features (X) and Target (y)
X = df.drop(columns=['Dataset'])
# Encode Target: Convert 1/2 classes to 0/1 (XGBoost prefers 0/1)
y = df['Dataset'].apply(lambda x: 1 if x == 1 else 0)

# C. Split Data (Using a fixed split due to the small size)
# NOTE: The data size is too small for a meaningful split and evaluation.
# This is for demonstration purposes only.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# D. Scaling Numerical Features (Crucial for proper model training)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame (optional, but good practice for feature visibility)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)


# --- 3. Model Training (XGBoost Classifier) ---

# Initialize the XGBoost model
xgb_model = XGBClassifier(
    n_estimators=100,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Train the model
xgb_model.fit(X_train_scaled, y_train)

print("✅ XGBoost Model Training Complete!")

# --- 4. Evaluation ---

# Make predictions on the test set
y_pred = xgb_model.predict(X_test_scaled)

print("\n--- Model Evaluation ---")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))