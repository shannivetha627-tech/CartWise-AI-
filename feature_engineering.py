
import pandas as pd
from pathlib import Path

# Load cleaned dataset
input_file = "data/processed/cleaned_retail.csv"
df = pd.read_csv(input_file)

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Output folders
processed_dir = Path("data/processed")
reports_dir = Path("data/reports")

processed_dir.mkdir(parents=True, exist_ok=True)
reports_dir.mkdir(parents=True, exist_ok=True)

print("========== FEATURE ENGINEERING ==========")

# 1. Create product description lookup
product_map = (
    df.dropna(subset=["Description"])
      .groupby("StockCode")["Description"]
      .agg(lambda x: x.mode().iloc[0])
)

product_map = product_map.reset_index()
product_map.columns = ["StockCode", "Product_Description"]

product_map.to_csv(
    processed_dir / "product_description_lookup.csv",
    index=False
)

# 2. Aggregate products within each invoice
# Sum quantities when a product appears more than once
invoice_products = (
    df.groupby(["InvoiceNo", "StockCode"])["Quantity"]
      .sum()
      .reset_index()
)

# Retain products with positive total quantities
invoice_products = invoice_products[
    invoice_products["Quantity"] > 0
].copy()

# 3. Generate product baskets
invoice_products.to_csv(
    processed_dir / "product_baskets.csv",
    index=False
)

# 4. Generate Boolean basket matrix
basket_matrix = (
    invoice_products
    .assign(Purchased=True)
    .pivot(
        index="InvoiceNo",
        columns="StockCode",
        values="Purchased"
    )
    .fillna(False)
    .astype(bool)
)

basket_matrix.to_csv(
    processed_dir / "basket_matrix.csv"
)

# 5. Calculate basket size
basket_size = basket_matrix.sum(axis=1)

basket_summary = pd.DataFrame({
    "InvoiceNo": basket_size.index,
    "Basket_Size": basket_size.values
})

basket_summary.to_csv(
    reports_dir / "basket_size_report.csv",
    index=False
)

# 6. Product purchase frequency
product_frequency = (
    invoice_products.groupby("StockCode")["InvoiceNo"]
    .nunique()
    .sort_values(ascending=False)
    .reset_index()
)

product_frequency.columns = [
    "StockCode",
    "Invoice_Frequency"
]

product_frequency = product_frequency.merge(
    product_map,
    on="StockCode",
    how="left"
)

product_frequency.to_csv(
    reports_dir / "product_frequency_report.csv",
    index=False
)

# 7. Feature summary
print("\nTotal invoices:", len(basket_matrix))
print("Unique products:", len(basket_matrix.columns))
print("Total product combinations:", len(invoice_products))

print("\nBasket size statistics:")
print(basket_size.describe())

print(
    "\nSingle-item baskets:",
    int((basket_size == 1).sum())
)

print(
    "Multi-item baskets:",
    int((basket_size >= 2).sum())
)

print("\nTop 10 frequently purchased products:")
print(product_frequency.head(10).to_string(index=False))

print("\nFeature engineering completed!")

print("\nGenerated files:")
print("data/processed/product_baskets.csv")
print("data/processed/basket_matrix.csv")
print("data/processed/product_description_lookup.csv")
print("data/reports/basket_size_report.csv")
print("data/reports/product_frequency_report.csv")
