
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load processed data
cleaned_file = "data/processed/cleaned_retail.csv"
basket_file = "data/reports/basket_size_report.csv"
frequency_file = "data/reports/product_frequency_report.csv"

df = pd.read_csv(cleaned_file)
basket_size = pd.read_csv(basket_file)
product_frequency = pd.read_csv(frequency_file)

# Create visualization folder
output_dir = Path("visualizations")
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Top 10 frequently purchased products
top_products = product_frequency.head(10)

plt.figure(figsize=(10, 6))
plt.bar(
    top_products["StockCode"].astype(str),
    top_products["Invoice_Frequency"]
)
plt.title("Top 10 Frequently Purchased Products")
plt.xlabel("Product Code")
plt.ylabel("Number of Invoices")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / "top_10_products.png")
plt.close()

# 2. Basket size distribution
plt.figure(figsize=(10, 6))
plt.hist(basket_size["Basket_Size"], bins=15, edgecolor="black")
plt.title("Shopping Basket Size Distribution")
plt.xlabel("Number of Unique Products per Invoice")
plt.ylabel("Number of Invoices")
plt.tight_layout()
plt.savefig(output_dir / "basket_size_distribution.png")
plt.close()

# 3. Quantity distribution
plt.figure(figsize=(10, 6))
plt.hist(df["Quantity"], bins=30, edgecolor="black")
plt.title("Product Quantity Distribution")
plt.xlabel("Quantity Purchased")
plt.ylabel("Number of Transaction Rows")
plt.tight_layout()
plt.savefig(output_dir / "quantity_distribution.png")
plt.close()

# 4. Product price distribution
plt.figure(figsize=(10, 6))
plt.hist(df["UnitPrice"], bins=30, edgecolor="black")
plt.title("Product Unit Price Distribution")
plt.xlabel("Unit Price")
plt.ylabel("Number of Transaction Rows")
plt.tight_layout()
plt.savefig(output_dir / "unit_price_distribution.png")
plt.close()

print("VISUALIZATION COMPLETED")
print("Total transaction rows:", len(df))
print("Total invoices:", df["InvoiceNo"].nunique())
print("Total unique products:", df["StockCode"].nunique())
print("\nCharts saved in the visualizations folder:")
for file in output_dir.glob("*.png"):
    print("-", file)
