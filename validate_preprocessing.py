
import pandas as pd
from pathlib import Path

# Load processed dataset
file_path = "data/processed/cleaned_retail.csv"
df = pd.read_csv(file_path)

# Create report folder
report_dir = Path("data/reports")
report_dir.mkdir(parents=True, exist_ok=True)

print("CARTWISE AI - PREPROCESSING VALIDATION")
print("=" * 45)

# 1. Check required columns
required_columns = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

# 2. Check missing values
missing_values = df[required_columns].isnull().sum()

# 3. Check invalid transaction values
invalid_quantity = (df["Quantity"] <= 0).sum()
invalid_price = (df["UnitPrice"] <= 0).sum()

# 4. Check duplicate rows
duplicate_rows = df.duplicated().sum()

# 5. Check product and invoice counts
unique_products = df["StockCode"].nunique()
unique_invoices = df["InvoiceNo"].nunique()

# 6. Check transaction date validity
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    errors="coerce"
)
invalid_dates = df["InvoiceDate"].isna().sum()

# 7. Feature selection for association-rule mining
selected_features = [
    "InvoiceNo",
    "StockCode",
    "Quantity"
]

# Create validation summary
validation_results = {
    "Total_Rows": len(df),
    "Total_Columns": len(df.columns),
    "Missing_Required_Columns": len(missing_columns),
    "Missing_Values_In_Required_Columns": int(
        missing_values.sum()
    ),
    "Invalid_Quantity_Rows": int(invalid_quantity),
    "Invalid_Price_Rows": int(invalid_price),
    "Duplicate_Rows": int(duplicate_rows),
    "Invalid_Date_Rows": int(invalid_dates),
    "Unique_Products": int(unique_products),
    "Unique_Invoices": int(unique_invoices),
    "Selected_Features": ", ".join(selected_features)
}

# Save validation report
report = pd.DataFrame(
    list(validation_results.items()),
    columns=["Validation_Check", "Result"]
)

report.to_csv(
    report_dir / "preprocessing_validation_report.csv",
    index=False
)

# Display results
for check, result in validation_results.items():
    print(f"{check}: {result}")

print("\nMissing values by required column:")
print(missing_values)

print("\nMissing required columns:")
print(missing_columns if missing_columns else "None")

print("\nReport saved:")
print(report_dir / "preprocessing_validation_report.csv")
