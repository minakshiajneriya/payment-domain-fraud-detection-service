# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Display settings
pd.set_option('display.max_columns', None)

# Import necessary libraries
import pandas as pd

fileName = "OriginalDataset.csv"

# Load the dataset (replace with your file path)
df = pd.read_csv(fileName)


# Check for missing values
# Drop columns with more than 30% missing values
#threshold = 0.3  # Set threshold
df = df.dropna(axis=1, thresh=int(threshold * len(df)))

# Impute missing numerical values with median
# Fill missing merchant_zipcode with the mode
# Drop rows with missing merchant_zipcode
# Replace missing merchant_zipcode with a placeholder
df.fillna({'merch_zipcode': 99999}, inplace=True)

df.to_csv('Cleaned_Dataset.csv', index=False)
print("Data cleaning completed and saved!")


# Boxplot to visualize outliers in transaction amount
plt.figure(figsize=(10, 5))
sns.boxplot(x=df['amt'])
plt.title('Boxplot of Transaction Amount')
plt.show()

# Compute IQR
Q1 = df['amt'].quantile(0.25)  # 25th percentile
Q3 = df['amt'].quantile(0.75)  # 75th percentile
IQR = Q3 - Q1  # Interquartile range

# Define lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find outliers
outliers = df[(df['amt'] < lower_bound) | (df['amt'] > upper_bound)]
print(f"Total outliers detected: {len(outliers)}")

# Visualize outliers vs fraud cases
plt.figure(figsize=(12, 6))
sns.scatterplot(x='amt', y='is_fraud', hue='amt', data=df)
plt.title('Outliers vs Fraud Cases')
plt.show()

# Cap the transaction amount at the 99.5th percentile
cap_value = df['amt'].quantile(0.995)
df['amt'] = df['amt'].clip(upper=cap_value)


