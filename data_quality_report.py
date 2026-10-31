
import pandas as pd
from pathlib import Path

# Load dataset
file_path = "data/raw/CartWise_AI_Synthetic_Dataset.csv"
df = pd.read_csv(file_path)

# Create quality report
report = pd.DataFrame({
    "Column": df.columns,
    "Data_Type": [str(df[col].dtype) for col in df.columns],
    "Missing_Values": [df[col].isnull().sum() for col in df.columns],
    "Missing_Percentage": [
        round(df[col].isnull().mean() * 100, 2)
        for col in df.columns
    ],
    "Unique_Values": [df[col].nunique() for col in df.columns]
})

# Save report
output_path = Path("data/reports")
output_path.mkdir(parents=True, exist_ok=True)

report.to_csv(
    output_path / "data_quality_report.csv",
    index=False
)

# Display results
print("\n========== DATA QUALITY REPORT ==========")
print(report.to_string(index=False))

print("\nTotal Rows:", len(df))
print("Total Columns:", len(df.columns))
print("Duplicate Rows:", df.duplicated().sum())

print("\nReport saved successfully!")
print("Location: data/reports/data_quality_report.csv")
