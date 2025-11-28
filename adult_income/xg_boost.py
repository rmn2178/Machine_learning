import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- 1. Initialize and Train the XGBoost Model ---

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_train = train.drop('income', axis="columns")
y_train = train['income']
X_test = test.drop('income', axis="columns")
y_test = test['income']

xgb_model = XGBClassifier(
    n_estimators=300,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=-1 # Utilize all cores for faster training
)

# Train the model directly on the preprocessed, numerical data
# XGBoost is highly effective on scaled and one-hot encoded features
xgb_model.fit(X_train, y_train)

print("🚀 XGBoost Model Training Complete!")

# --- 2. Make Predictions and Evaluate ---

# Make predictions on the test set
y_pred_xgb = xgb_model.predict(X_test)

print("\n--- XGBoost Model Evaluation ---")
print(f"Accuracy on Test Set: {accuracy_score(y_test, y_pred_xgb):.4f}")

# Detailed Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))