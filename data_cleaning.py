
import pandas as pd
from pathlib import Path

# Load original synthetic dataset
input_file = "data/raw/CartWise_AI_Synthetic_Dataset.csv"
df = pd.read_csv(input_file)

print("Original dataset shape:", df.shape)

# Cleaning audit
audit = []

def record_step(name, before, after):
    audit.append({
        "Cleaning_Step": name,
        "Rows_Before": before,
        "Rows_After": after,
        "Rows_Removed": before - after
    })

# 1. Remove exact duplicate rows
before = len(df)
df = df.drop_duplicates().copy()
record_step("Duplicate removal", before, len(df))

# 2. Standardize product codes
df["StockCode"] = (
    df["StockCode"]
    .astype("string")
    .str.strip()
    .str.upper()
)

# 3. Standardize product descriptions
df["Description"] = (
    df["Description"]
    .astype("string")
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# 4. Identify cancellation invoices
before = len(df)
df = df[
    ~df["InvoiceNo"].astype("string").str.startswith("C", na=False)
].copy()
record_step("Cancellation removal", before, len(df))

# 5. Exclude non-product StockCodes
non_product_codes = {
    "POST", "D", "M", "DOT", "BANK CHARGES",
    "CRUK", "AMAZONFEE", "S", "B"
}

before = len(df)
df = df[~df["StockCode"].isin(non_product_codes)].copy()
record_step("Non-product code removal", before, len(df))

# 6. Remove invalid quantities
before = len(df)
df = df[df["Quantity"] > 0].copy()
record_step("Invalid quantity removal", before, len(df))

# 7. Remove invalid prices
before = len(df)
df = df[df["UnitPrice"] > 0].copy()
record_step("Invalid price removal", before, len(df))

# 8. Fill missing descriptions using the most frequent
# description associated with each StockCode
description_map = (
    df.dropna(subset=["Description"])
      .groupby("StockCode")["Description"]
      .agg(lambda values: values.mode().iloc[0])
)

before = len(df)
df["Description"] = df["Description"].fillna(
    df["StockCode"].map(description_map)
)

# Drop only rows whose descriptions remain missing
df = df.dropna(subset=["Description"]).copy()
record_step("Unresolved description removal", before, len(df))

# 9. Convert dates
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    errors="coerce"
)

before = len(df)
df = df.dropna(subset=["InvoiceDate"]).copy()
record_step("Invalid date removal", before, len(df))

# 10. Flag extreme quantities for investigation
# Do not automatically delete valid large orders
quantity_99 = df["Quantity"].quantile(0.99)
df["Quantity_Outlier_Flag"] = df["Quantity"] > quantity_99

# Save cleaned dataset
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

df.to_csv(
    output_dir / "cleaned_retail.csv",
    index=False
)

# Save audit report
audit_df = pd.DataFrame(audit)
audit_df.to_csv(
    "data/reports/cleaning_audit.csv",
    index=False
)

print("\n========== CLEANING RESULTS ==========")
print("Original rows:", sum(
    [audit[0]["Rows_Before"]]
))
print("Cleaned rows:", len(df))
print("Remaining missing values:")
print(df.isnull().sum())

print("\nCleaning audit:")
print(audit_df.to_string(index=False))

print("\nQuantity 99th percentile:", quantity_99)
print("Flagged quantity outliers:", df["Quantity_Outlier_Flag"].sum())

print("\nCleaned dataset saved:")
print("data/processed/cleaned_retail.csv")
print("Audit saved:")
print("data/reports/cleaning_audit.csv")
