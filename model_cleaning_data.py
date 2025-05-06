import pandas as pd

# Load the dataset
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

# ✅ Identify non-numeric columns
non_numeric_cols = X_train.select_dtypes(exclude=['number']).columns
print("Non-numeric columns:", non_numeric_cols)

# ✅ Drop non-numeric columns
X_train = X_train.drop(non_numeric_cols, axis=1)
X_test = X_test.drop(non_numeric_cols, axis=1)

# ✅ Save the cleaned CSV files
X_train.to_csv("X_train_clean.csv", index=False)
X_test.to_csv("X_test_clean.csv", index=False)

print("Non-numeric columns removed and saved as cleaned CSVs!")
