
import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

RULES_FILE = BASE_DIR / "data/reports/apriori_association_rules.csv"
PRICES_FILE = BASE_DIR / "data/processed/product_price_lookup.csv"


def parse_products(value):
    return set(re.findall(r"P\d+", str(value)))


def load_data():
    rules = pd.read_csv(RULES_FILE)
    prices = pd.read_csv(PRICES_FILE)

    required_rules = {
        "antecedents", "consequents",
        "support", "confidence", "lift"
    }
    required_prices = {"StockCode", "Description", "UnitPrice"}

    if not required_rules.issubset(rules.columns):
        raise ValueError("Missing association rule columns.")

    if not required_prices.issubset(prices.columns):
        raise ValueError("Missing product price columns.")

    if prices["StockCode"].duplicated().any():
        raise ValueError("Duplicate product IDs found.")

    if prices["UnitPrice"].isna().any() or (prices["UnitPrice"] <= 0).any():
        raise ValueError("Invalid product prices found.")

    rules["antecedent_items"] = rules["antecedents"].apply(parse_products)
    rules["consequent_items"] = rules["consequents"].apply(parse_products)

    return rules, prices


def recommend_products(selected_products, budget, max_recommendations=5):
    rules, prices = load_data()

    budget = float(budget)

    if budget < 0:
        raise ValueError("Budget cannot be negative.")

    if max_recommendations < 1:
        raise ValueError("Maximum recommendations must be positive.")

    selected = {str(p).strip().upper() for p in selected_products}

    known_products = set(prices["StockCode"].astype(str))
    unknown = selected - known_products

    if unknown:
        raise ValueError(f"Unknown product IDs: {sorted(unknown)}")

    candidates = {}

    matching_rules = rules[
        rules["antecedent_items"].apply(
            lambda items: bool(items) and items.issubset(selected)
        )
    ]

    for row in matching_rules.itertuples():
        for product_id in row.consequent_items - selected:
            score = (
                float(row.confidence),
                float(row.lift),
                float(row.support)
            )

            if product_id not in candidates or score > candidates[product_id]:
                candidates[product_id] = score

    product_lookup = prices.set_index("StockCode").to_dict("index")

    ranked = []

    for product_id, score in candidates.items():
        product = product_lookup[product_id]

        ranked.append({
            "StockCode": product_id,
            "Description": product["Description"],
            "UnitPrice": float(product["UnitPrice"]),
            "Confidence": score[0],
            "Lift": score[1],
            "Support": score[2]
        })

    ranked.sort(
        key=lambda x: (
            -x["Confidence"],
            -x["Lift"],
            -x["Support"],
            x["UnitPrice"],
            x["StockCode"]
        )
    )

    recommendations = []
    remaining_budget = budget

    for product in ranked:
        if len(recommendations) >= max_recommendations:
            break

        if product["UnitPrice"] <= remaining_budget + 1e-9:
            recommendations.append(product)
            remaining_budget -= product["UnitPrice"]

    result = pd.DataFrame(
        recommendations,
        columns=[
            "StockCode", "Description", "UnitPrice",
            "Confidence", "Lift", "Support"
        ]
    )

    summary = {
        "budget": budget,
        "recommended_cost": budget - remaining_budget,
        "remaining_budget": max(0.0, remaining_budget),
        "recommendation_count": len(result)
    }

    return result, summary


if __name__ == "__main__":
    print("=" * 60)
    print("CARTWISE AI - BUDGET-AWARE RECOMMENDATION ENGINE")
    print("=" * 60)

    selected_products = ["P003"]
    available_budget = 5.00

    recommendations, summary = recommend_products(
        selected_products,
        available_budget
    )

    print("\nSelected products:", selected_products)
    print("Additional budget:", f"{available_budget:.2f}")

    print("\nRECOMMENDED PRODUCTS")

    if recommendations.empty:
        print("No associated products fit the available budget.")
    else:
        print(recommendations.to_string(index=False))

    print("\nBUDGET SUMMARY")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nRECOMMENDATION ENGINE COMPLETED")
