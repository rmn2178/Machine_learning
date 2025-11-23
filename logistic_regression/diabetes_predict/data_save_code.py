import pandas as pd

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome" # This is the target variable (0 or 1)
]

df = pd.read_csv(url, header=None, names=columns)

print("--- DataFrame with Headings ---")
print(df.head()) # Check the first few rows to confirm headers are present
print("-" * 35)

# 3. Save the DataFrame to a new CSV file
output_file_name = 'diabetes.csv'

# When you save with to_csv(), Pandas automatically includes the column headers
# at the top unless you specify header=False (which we don't do here).
# index=False ensures the row numbers aren't included as a column.
df.to_csv(output_file_name, index=False)

print(f"✅ Data successfully loaded, titled, and saved to {output_file_name}")