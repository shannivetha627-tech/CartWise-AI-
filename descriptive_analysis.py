
import pandas as pd
from pathlib import Path

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_retail.csv")

# Convert date column
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Create output folder
report_dir = Path("data/reports")
report_dir.mkdir(parents=True, exist_ok=True)

print("=" * 55)
print("CARTWISE AI - DESCRIPTIVE DATA ANALYSIS")
print("=" * 55)

# 1. Dataset overview
overview = {
    "Total_Transaction_Rows": len(df),
    "Unique_Invoices": df["InvoiceNo"].nunique(),
    "Unique_Products": df["StockCode"].nunique(),
    "Unique_Customers": df["CustomerID"].nunique(),
    "Unique_Countries": df["Country"].nunique()
}

print("\n1. DATASET OVERVIEW")
for key, value in overview.items():
    print(f"{key}: {value}")

# 2. Quantity statistics
quantity_stats = df["Quantity"].describe()

print("\n2. QUANTITY STATISTICS")
print(quantity_stats)

# 3. Unit price statistics
price_stats = df["UnitPrice"].describe()

print("\n3. UNIT PRICE STATISTICS")
print(price_stats)

# 4. Invoice-level basket statistics
basket_sizes = (
    df.groupby("InvoiceNo")["StockCode"]
      .nunique()
)

basket_stats = basket_sizes.describe()

print("\n4. BASKET SIZE STATISTICS")
print(basket_stats)

# 5. Top 10 frequently purchased products
top_products = (
    df.groupby("StockCode")["InvoiceNo"]
      .nunique()
      .sort_values(ascending=False)
      .head(10)
      .reset_index(name="Invoice_Frequency")
)

print("\n5. TOP 10 PRODUCTS")
print(top_products.to_string(index=False))

# 6. Country-level transaction analysis
country_analysis = (
    df.groupby("Country")
      .agg(
          Transaction_Rows=("InvoiceNo", "count"),
          Unique_Invoices=("InvoiceNo", "nunique"),
          Unique_Products=("StockCode", "nunique")
      )
      .sort_values("Unique_Invoices", ascending=False)
      .reset_index()
)

print("\n6. COUNTRY ANALYSIS")
print(country_analysis.to_string(index=False))

# 7. Save reports
overview_df = pd.DataFrame(
    list(overview.items()),
    columns=["Metric", "Value"]
)

overview_df.to_csv(
    report_dir / "dataset_overview.csv",
    index=False
)

quantity_stats.to_csv(
    report_dir / "quantity_statistics.csv"
)

price_stats.to_csv(
    report_dir / "price_statistics.csv"
)

basket_stats.to_csv(
    report_dir / "basket_statistics.csv"
)

top_products.to_csv(
    report_dir / "top_products_analysis.csv",
    index=False
)

country_analysis.to_csv(
    report_dir / "country_analysis.csv",
    index=False
)

print("\n" + "=" * 55)
print("DESCRIPTIVE ANALYSIS COMPLETED")
print("Reports saved in data/reports/")
print("=" * 55)
