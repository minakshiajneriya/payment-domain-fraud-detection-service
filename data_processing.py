# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Display settings
pd.set_option('display.max_columns', None)


fileName = "Cleaned_Dataset.csv"

# Load the dataset (replace with your file path)
# 
df = pd.read_csv(fileName)

import pandas as pd
import numpy as np

# Ensure the datetime column is in proper format
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])

# 1️⃣ Extracting Time-Based Features
df['year'] = df['trans_date_trans_time'].dt.year
df['month'] = df['trans_date_trans_time'].dt.month
df['day'] = df['trans_date_trans_time'].dt.day
df['day_of_week'] = df['trans_date_trans_time'].dt.weekday
df['hour'] = df['trans_date_trans_time'].dt.hour
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# 2️⃣ Transaction Velocity Features
# Sorting by card number and transaction time
df = df.sort_values(['cc_num', 'trans_date_trans_time'])

# ✅ Transactions in the last 1 hour
df['transactions_last_1hr'] = (
    df.set_index('trans_date_trans_time')
    .groupby('cc_num')['cc_num']
    .rolling('1H')
    .count()
    .reset_index(drop=True) - 1
)

# ✅ Transactions in the last 24 hours
df['transactions_last_24hr'] = (
    df.set_index('trans_date_trans_time')
    .groupby('cc_num')['cc_num']
    .rolling('24H')
    .count()
    .reset_index(drop=True) - 1
)

# ✅ Average amount in the last 24 hours
df['avg_amt_last_24hr'] = (
    df.set_index('trans_date_trans_time')
    .groupby('cc_num')['amt']
    .rolling('24H')
    .mean()
    .reset_index(drop=True)
)

# ✅ Replace inplace with direct assignment
df['transactions_last_1 hr'] = df['transactions_last_1hr'].fillna(0)
df['transactions_last_24hr'] = df['transactions_last_24hr'].fillna(0)
df['avg_amt_last_24hr'] = df['avg_amt_last_24hr'].fillna(0)

# ✅ Final dataframe with new features
#print(df[['cc_num', 'trans_date_trans_time', 'transactions_last_1hr', 'transactions_last_24hr', 'avg_amt_last_24hr']].head())


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 🚀 Load the dataset
# df = pd.read_csv('your_cleaned_data.csv')  # Use your cleaned dataset

# ✅ List of categorical columns
categorical_cols = ['gender', 'category', 'state', 'merchant', 'job']

# 💡 Apply Label Encoding for binary categories (e.g., gender)
label_encoder = LabelEncoder()
df['gender'] = label_encoder.fit_transform(df['gender'])

# 🛠️ Apply One-Hot Encoding for multi-class categories
df = pd.get_dummies(df, columns=['category', 'state', 'merchant', 'job'], drop_first=True)

# 🧐 Display the transformed data
#print("\nEncoded Dataset:")
#print(df.head())


###########################Feature Scaling###################
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ✅ List of numerical columns to scale
num_cols = ['amt', 'transactions_last_1hr', 'transactions_last_24hr', 'avg_amt_last_24hr']

# 💡 Choose the scaling method
# MinMax Scaling (0 to 1)
minmax_scaler = MinMaxScaler()
df[num_cols] = minmax_scaler.fit_transform(df[num_cols])

# Standard Scaling (mean = 0, std = 1) - Uncomment if you prefer this
# standard_scaler = StandardScaler()
# df[num_cols] = standard_scaler.fit_transform(df[num_cols])

# ✅ Verify the scaled data
#print("\nScaled Dataset:")
#print(df[num_cols].head())

from sklearn.model_selection import train_test_split

# ✅ Define features and target
X = df.drop('is_fraud', axis=1)   # Features
y = df['is_fraud']                 # Target variable

# ✅ Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,                  # 80% training, 20% testing
    random_state=42,                # Ensures reproducibility
    stratify=y                      # Maintain fraud ratio in both sets
)

# ✅ Print the shape of the splits
#print(f"Training set: {X_train.shape}, {y_train.shape}")
#print(f"Testing set: {X_test.shape}, {y_test.shape}")

# Check the fraud ratio in both sets
#print("\nClass distribution in Training Set:")
#print(y_train.value_counts(normalize=True))

#print("\nClass distribution in Testing Set:")
#print(y_test.value_counts(normalize=True))

# Save the split data
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("✅ Data splitting completed and saved!")

