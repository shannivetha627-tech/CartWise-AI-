
import pandas as pd
import numpy as np
from pathlib import Path

# Reproducibility
rng = np.random.default_rng(42)

# Output directory
output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

# Product catalogue
products = {
    "P001": ("WHITE HANGING HEART T-LIGHT HOLDER", 2.95),
    "P002": ("RED WOOLLY HOTTIE WHITE HEART", 3.75),
    "P003": ("JUMBO BAG RED RETROSPOT", 1.95),
    "P004": ("JUMBO BAG PINK POLKADOT", 1.95),
    "P005": ("LUNCH BAG RED RETROSPOT", 1.65),
    "P006": ("LUNCH BAG BLACK SKULL", 1.65),
    "P007": ("REGENCY CAKESTAND 3 TIER", 12.75),
    "P008": ("PARTY BUNTING", 4.95),
    "P009": ("PAPER CHAIN KIT 50'S CHRISTMAS", 2.55),
    "P010": ("PACK OF 72 RETROSPOT CAKE CASES", 0.55),
    "P011": ("HEART OF WICKER SMALL", 1.65),
    "P012": ("HEART OF WICKER LARGE", 2.95),
    "P013": ("CERAMIC STRAWBERRY CAKE MONEY BANK", 1.25),
    "P014": ("WOODEN PICTURE FRAME", 2.50),
    "P015": ("SET OF 3 CAKE TINS", 4.95),
    "P016": ("RETRO COFFEE MUG", 3.25),
    "P017": ("BLUE CERAMIC MUG", 2.75),
    "P018": ("RED CERAMIC MUG", 2.75),
    "P019": ("GLASS STAR T-LIGHT HOLDER", 1.25),
    "P020": ("VINTAGE CHRISTMAS STOCKING", 3.95),
    "P021": ("ASSORTED COLOUR BIRD ORNAMENT", 1.69),
    "P022": ("PINK REGENCY TEACUP", 2.95),
    "P023": ("GREEN REGENCY TEACUP", 2.95),
    "P024": ("SET OF 6 SPICE TINS", 3.95),
    "P025": ("WOODEN HEART DECORATION", 0.85),
    "P026": ("SMALL GLASS CANDLE", 1.25),
    "P027": ("LARGE GLASS CANDLE", 2.50),
    "P028": ("FLORAL GIFT WRAP", 1.50),
    "P029": ("VINTAGE GIFT BAG", 2.10),
    "P030": ("CERAMIC STORAGE JAR", 4.25),
    "P031": ("KITCHEN APRON", 5.95),
    "P032": ("TEA TOWEL SET", 3.50),
    "P033": ("METAL SIGN", 2.95),
    "P034": ("DECORATIVE PLANT POT", 4.50),
    "P035": ("GARDEN GLOVE SET", 3.25),
    "P036": ("BIRTHDAY CARD SET", 1.95),
    "P037": ("CHRISTMAS CARD SET", 2.25),
    "P038": ("DECORATIVE RIBBON", 1.15),
    "P039": ("COTTON SHOPPING BAG", 2.50),
    "P040": ("WOODEN TOY SET", 6.95),
}

# Product bundles create learnable co-purchase patterns
bundles = [
    ["P001", "P019", "P026", "P027"],
    ["P003", "P004", "P005", "P006"],
    ["P007", "P010", "P015"],
    ["P008", "P009", "P020", "P037"],
    ["P011", "P012", "P025", "P033"],
    ["P016", "P017", "P018", "P022", "P023"],
    ["P028", "P029", "P036", "P038"],
    ["P031", "P032", "P024", "P030"],
    ["P034", "P035", "P039"],
    ["P013", "P014", "P021", "P040"],
]

countries = [
    "United Kingdom", "Germany", "France",
    "Spain", "Netherlands", "EIRE"
]

rows = []

# Generate 2,000 invoices with 5 product lines each
for invoice_id in range(1, 2001):

    bundle = bundles[(invoice_id - 1) % len(bundles)]

    selected = list(bundle)

    # Fill each basket to exactly five distinct products
    remaining = [
        code for code in products if code not in selected
    ]

    extra_count = 5 - len(selected)

    if extra_count > 0:
        selected.extend(
            rng.choice(
                remaining,
                size=extra_count,
                replace=False
            ).tolist()
        )

    date = (
        pd.Timestamp("2025-01-01")
        + pd.Timedelta(days=int(rng.integers(0, 365)))
        + pd.Timedelta(hours=int(rng.integers(8, 21)))
        + pd.Timedelta(minutes=int(rng.integers(0, 60)))
    )

    customer_id = int(rng.integers(10000, 11001))
    country = rng.choice(countries, p=[0.65, 0.10, 0.08, 0.06, 0.06, 0.05])

    is_cancelled = rng.random() < 0.04

    invoice_number = f"{invoice_id:06d}"

    if is_cancelled:
        invoice_number = "C" + invoice_number

    for code in selected:
        description, price = products[code]

        quantity = int(rng.integers(1, 6))

        if is_cancelled:
            quantity = -quantity

        rows.append({
            "InvoiceNo": invoice_number,
            "StockCode": code,
            "Description": description,
            "Quantity": quantity,
            "InvoiceDate": date,
            "UnitPrice": price,
            "CustomerID": customer_id,
            "Country": country
        })

df = pd.DataFrame(rows)

# Inject realistic data-quality problems for cleaning practice

# Missing customer IDs
missing_ids = rng.choice(df.index, 300, replace=False)
df.loc[missing_ids, "CustomerID"] = np.nan

# Missing descriptions
missing_desc = rng.choice(df.index, 80, replace=False)
df.loc[missing_desc, "Description"] = np.nan

# Duplicate records
duplicates = df.sample(100, random_state=42)
df = pd.concat([df, duplicates], ignore_index=True)

# Zero-price records
zero_price = rng.choice(df.index, 30, replace=False)
df.loc[zero_price, "UnitPrice"] = 0

# Negative-price records
negative_price = rng.choice(df.index, 5, replace=False)
df.loc[negative_price, "UnitPrice"] = -1

# Extreme quantity records
outliers = rng.choice(df.index, 5, replace=False)
df.loc[outliers, "Quantity"] = 80995

# Add service/accounting codes
service_rows = []

service_codes = [
    ("POST", "POSTAGE"),
    ("D", "DISCOUNT"),
    ("M", "MANUAL"),
    ("DOT", "DOTCOM POSTAGE"),
    ("BANK CHARGES", "BANK CHARGES")
]

for i in range(50):
    code, description = service_codes[i % len(service_codes)]

    service_rows.append({
        "InvoiceNo": f"{2001 + i:06d}",
        "StockCode": code,
        "Description": description,
        "Quantity": 1,
        "InvoiceDate": pd.Timestamp("2025-06-01"),
        "UnitPrice": 2.50,
        "CustomerID": 10500,
        "Country": "United Kingdom"
    })

df = pd.concat(
    [df, pd.DataFrame(service_rows)],
    ignore_index=True
)

# Mix record order
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save CSV
output_path = output_dir / "CartWise_AI_Synthetic_Dataset.csv"
df.to_csv(output_path, index=False)

print("CartWise AI synthetic dataset generated!")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print("Saved to:", output_path)
