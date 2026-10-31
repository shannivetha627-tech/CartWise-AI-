
import pandas as pd
from pathlib import Path

# File paths
cleaned_file = Path("data/processed/cleaned_retail.csv")
basket_file = Path("data/processed/product_baskets.csv")
matrix_file = Path("data/processed/basket_matrix.csv")
quality_file = Path("data/reports/data_quality_report.csv")
audit_file = Path("data/reports/cleaning_audit.csv")
validation_file = Path(
    "data/reports/preprocessing_validation_report.csv"
)

# Check required files
required_files = [
    cleaned_file,
    basket_file,
    matrix_file,
    quality_file,
    audit_file,
    validation_file
]

missing_files = [str(file) for file in required_files if not file.exists()]

if missing_files:
    print("Some required files are missing:")
    for file in missing_files:
        print("-", file)
    raise SystemExit("Complete the previous steps before continuing.")

# Load data
df = pd.read_csv(cleaned_file)
baskets = pd.read_csv(basket_file)
basket_matrix = pd.read_csv(matrix_file, index_col=0)

# Calculate summary
summary = {
    "Project": "CartWise AI",
    "Week": "Week 2",
    "Dataset_Type": "Synthetic retail transaction dataset",
    "Cleaned_Transaction_Rows": len(df),
    "Unique_Invoices": df["InvoiceNo"].nunique(),
    "Unique_Products": df["StockCode"].nunique(),
    "Product_Basket_Rows": len(baskets),
    "Basket_Matrix_Rows": basket_matrix.shape[0],
    "Basket_Matrix_Columns": basket_matrix.shape[1],
    "Missing_Values_In_Cleaned_Data": int(df.isnull().sum().sum()),
    "Duplicate_Rows_In_Cleaned_Data": int(df.duplicated().sum())
}

# Save summary
report_dir = Path("data/reports")
report_dir.mkdir(parents=True, exist_ok=True)

summary_df = pd.DataFrame(
    list(summary.items()),
    columns=["Metric", "Result"]
)

summary_df.to_csv(
    report_dir / "week2_preprocessing_summary.csv",
    index=False
)

print("=" * 50)
print("CARTWISE AI - WEEK 2 SUMMARY")
print("=" * 50)

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nSummary saved to:")
print(report_dir / "week2_preprocessing_summary.csv")
