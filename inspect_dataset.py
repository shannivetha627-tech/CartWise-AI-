import pandas as pd

# Load synthetic dataset
file_path = "data/raw/CartWise_AI_Synthetic_Dataset.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET OVERVIEW ==========")

# Dataset dimensions
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns.tolist())

# First five records
print("\nFirst 5 Records:")
print(df.head())

# Data types and non-null counts
print("\nDataset Information:")
df.info()

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate records
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe(include="all"))