import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # <-- ADDED: For data scaling

# --- Data Loading and Splitting ---

# NOTE on Split: The order of variables (test_df, train_df)
# is swapped based on the test_size=0.1.
# Here, train_df is 10% and test_df is 90% of the data. This is unusual.
df = pd.read_csv("diabetes.csv")
test_df , train_df = train_test_split(df, test_size = 0.1, random_state = 42)

X_train = train_df.drop("Outcome", axis = 1)
Y_train = train_df["Outcome"]
X_test = test_df.drop("Outcome", axis = 1)
Y_test = test_df["Outcome"]

# --- ADDED: Data Scaling ---

# 1. Initialize the scaler
scaler = StandardScaler()

# 2. Fit the scaler ONLY on the training features and transform
X_train_scaled = scaler.fit_transform(X_train)

# 3. Transform the test features (DO NOT FIT on test data!)
X_test_scaled = scaler.transform(X_test)


# --- Model Training ---

regressor = LogisticRegression(max_iter=5000) # Increased max_iter for safety
regression = regressor.fit(X_train_scaled, Y_train) # <-- Use SCALED data

# Print model coefficients
print("\nModel Coefficients:")
print("Intercept:", regression.intercept_)
print("Coefficients:", regression.coef_)

# Predicting with the made model
pred = regression.predict(X_test_scaled) # <-- Predict using SCALED data

# Create a DataFrame to compare actual and predicted outcomes
results = pd.DataFrame({
    'Actual Outcome': Y_test.values, # Changed column name to reflect binary classification
    'Predicted Outcome': pred
})
print("\n--- Sample Prediction Results ---")
print(results.head())

# --- FIX: Accuracy Calculation ---

# ERROR line: accuracy = regression.score(pred, Y_test)
# FIX: The score method expects the FEATURES (X_test_scaled) and the true TARGET (Y_test).
accuracy = regression.score(X_test_scaled, Y_test)

print("\nAccuracy:", accuracy)