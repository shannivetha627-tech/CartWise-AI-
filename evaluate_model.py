
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


# ============================================================
# CARTWISE AI - WEEK 4
# TRAIN-TEST MODEL EVALUATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

BASKET_FILE = BASE_DIR / "data/processed/basket_matrix.csv"
PRICE_FILE = BASE_DIR / "data/processed/product_price_lookup.csv"

REPORT_DIR = BASE_DIR / "data/reports"
VISUALIZATION_DIR = BASE_DIR / "visualizations"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.30
MAX_ITEMSET_LENGTH = 3

BUDGET_LEVELS = [2.00, 5.00, 10.00, 15.00]
MAX_RECOMMENDATIONS = 5


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 65)
print("CARTWISE AI - WEEK 4 MODEL EVALUATION")
print("=" * 65)

basket = pd.read_csv(BASKET_FILE, index_col=0)
basket = basket.fillna(False).astype(bool)

prices = pd.read_csv(PRICE_FILE)

if basket.empty:
    raise ValueError("Basket dataset is empty.")

if prices["StockCode"].duplicated().any():
    raise ValueError("Duplicate product IDs found in price lookup.")

price_lookup = prices.set_index("StockCode")["UnitPrice"].to_dict()

if not set(basket.columns).issubset(price_lookup):
    raise ValueError("Some basket products have no matching price.")

print("\nDATASET INFORMATION")
print("-" * 40)
print("Total transactions:", len(basket))
print("Total products:", basket.shape[1])
print("Total unique basket patterns:", len(basket.drop_duplicates()))


# ============================================================
# 2. TRAIN-TEST SPLIT
# ============================================================

rng = np.random.default_rng(RANDOM_STATE)

indices = rng.permutation(len(basket))
test_count = int(round(len(basket) * TEST_SIZE))

test_indices = indices[:test_count]
train_indices = indices[test_count:]

train_data = basket.iloc[train_indices].reset_index(drop=True)
test_data = basket.iloc[test_indices].reset_index(drop=True)

print("\nTRAIN-TEST SPLIT")
print("-" * 40)
print("Training transactions:", len(train_data))
print("Testing transactions:", len(test_data))
print("Training percentage:", round(len(train_data) / len(basket) * 100, 2))
print("Testing percentage:", round(len(test_data) / len(basket) * 100, 2))


# ============================================================
# 3. MINE APRIORI RULES USING TRAINING DATA ONLY
# ============================================================

print("\nMINING ASSOCIATION RULES")
print("-" * 40)

frequent_itemsets = apriori(
    train_data,
    min_support=MIN_SUPPORT,
    use_colnames=True,
    max_len=MAX_ITEMSET_LENGTH,
    low_memory=True
)

if frequent_itemsets.empty:
    raise ValueError("No frequent itemsets found.")

rule_args = {
    "metric": "confidence",
    "min_threshold": MIN_CONFIDENCE
}

if "num_itemsets" in inspect.signature(association_rules).parameters:
    rule_args["num_itemsets"] = len(train_data)

rules = association_rules(frequent_itemsets, **rule_args)

if rules.empty:
    raise ValueError("No association rules found.")

rules = rules.sort_values(
    ["confidence", "lift", "support"],
    ascending=[False, False, False]
).reset_index(drop=True)

print("Frequent itemsets:", len(frequent_itemsets))
print("Association rules:", len(rules))


# ============================================================
# 4. EVALUATE RULES ON UNSEEN TEST DATA
# ============================================================

def calculate_test_rule_metrics(rule, test_matrix):
    """
    Calculate rule metrics using held-out test transactions.
    """

    antecedents = set(rule["antecedents"])
    consequents = set(rule["consequents"])

    combined_items = antecedents | consequents

    total = len(test_matrix)

    antecedent_count = int(
        test_matrix[list(antecedents)].all(axis=1).sum()
    )

    consequent_count = int(
        test_matrix[list(consequents)].all(axis=1).sum()
    )

    combined_count = int(
        test_matrix[list(combined_items)].all(axis=1).sum()
    )

    antecedent_support = antecedent_count / total
    consequent_support = consequent_count / total
    test_support = combined_count / total

    test_confidence = (
        combined_count / antecedent_count
        if antecedent_count > 0 else np.nan
    )

    test_lift = (
        test_confidence / consequent_support
        if consequent_support > 0 and not np.isnan(test_confidence)
        else np.nan
    )

    test_leverage = (
        test_support - (antecedent_support * consequent_support)
    )

    if np.isnan(test_confidence):
        test_conviction = np.nan
    elif test_confidence >= 1:
        test_conviction = np.inf
    elif consequent_support < 1:
        test_conviction = (
            (1 - consequent_support) / (1 - test_confidence)
        )
    else:
        test_conviction = np.nan

    return {
        "test_antecedent_count": antecedent_count,
        "test_support": test_support,
        "test_confidence": test_confidence,
        "test_lift": test_lift,
        "test_leverage": test_leverage,
        "test_conviction": test_conviction
    }


rule_results = []

for _, rule in rules.iterrows():
    metrics = calculate_test_rule_metrics(rule, test_data)

    rule_results.append({
        "antecedents": ", ".join(sorted(rule["antecedents"])),
        "consequents": ", ".join(sorted(rule["consequents"])),
        "train_support": rule["support"],
        "train_confidence": rule["confidence"],
        "train_lift": rule["lift"],
        **metrics
    })

rule_evaluation = pd.DataFrame(rule_results)

rule_evaluation.to_csv(
    REPORT_DIR / "train_test_rule_evaluation.csv",
    index=False
)

print("\nRULE EVALUATION")
print("-" * 40)
print("Rules evaluated:", len(rule_evaluation))
print(
    "Rules observed in test data:",
    int((rule_evaluation["test_antecedent_count"] > 0).sum())
)
print(
    "Mean test confidence:",
    round(rule_evaluation["test_confidence"].mean(), 4)
)
print(
    "Mean test lift:",
    round(rule_evaluation["test_lift"].mean(), 4)
)


# ============================================================
# 5. RECOMMENDATION GENERATION
# ============================================================

def generate_recommendations(selected_products, budget):
    """
    Reproduce the Week 3 recommendation ranking and
    budget-selection logic using training-derived rules.
    """

    selected = set(selected_products)
    candidates = {}

    matching_rules = rules[
        rules["antecedents"].apply(
            lambda items: bool(items) and set(items).issubset(selected)
        )
    ]

    for rule in matching_rules.itertuples():
        for product_id in set(rule.consequents) - selected:

            score = (
                float(rule.confidence),
                float(rule.lift),
                float(rule.support)
            )

            if (
                product_id not in candidates
                or score > candidates[product_id]
            ):
                candidates[product_id] = score

    ranked = []

    for product_id, score in candidates.items():

        price = float(price_lookup[product_id])

        ranked.append({
            "StockCode": product_id,
            "UnitPrice": price,
            "Confidence": score[0],
            "Lift": score[1],
            "Support": score[2]
        })

    ranked.sort(
        key=lambda item: (
            -item["Confidence"],
            -item["Lift"],
            -item["Support"],
            item["UnitPrice"],
            item["StockCode"]
        )
    )

    recommendations = []
    remaining_budget = float(budget)

    for product in ranked:

        if len(recommendations) >= MAX_RECOMMENDATIONS:
            break

        if product["UnitPrice"] <= remaining_budget + 1e-9:
            recommendations.append(product)
            remaining_budget -= product["UnitPrice"]

    return recommendations


# ============================================================
# 6. EVALUATE RECOMMENDATION QUALITY
# ============================================================

case_results = []
summary_results = []

for budget in BUDGET_LEVELS:

    all_precision = []
    all_recall = []
    all_hits = []

    total_cases = 0
    cases_with_recommendations = 0
    budget_violations = 0
    total_spent = 0.0

    for _, transaction in test_data.iterrows():

        actual_products = set(
            transaction[transaction].index
        )

        # Evaluate each product as the customer's selected product.
        for selected_product in sorted(actual_products):

            relevant_products = actual_products - {selected_product}

            recommendations = generate_recommendations(
                [selected_product],
                budget
            )

            recommended_products = {
                item["StockCode"] for item in recommendations
            }

            recommended_cost = sum(
                item["UnitPrice"] for item in recommendations
            )

            hits = recommended_products & relevant_products

            precision_at_5 = len(hits) / MAX_RECOMMENDATIONS

            recall_at_5 = (
                len(hits) / len(relevant_products)
                if relevant_products else 0.0
            )

            hit_rate = 1 if hits else 0

            total_cases += 1

            if recommendations:
                cases_with_recommendations += 1

            if recommended_cost > budget + 1e-9:
                budget_violations += 1

            total_spent += recommended_cost

            all_precision.append(precision_at_5)
            all_recall.append(recall_at_5)
            all_hits.append(hit_rate)

            case_results.append({
                "transaction_index": int(_),
                "selected_product": selected_product,
                "budget": budget,
                "actual_relevant_products": ", ".join(
                    sorted(relevant_products)
                ),
                "recommended_products": ", ".join(
                    sorted(recommended_products)
                ),
                "correct_recommendations": len(hits),
                "precision_at_5": precision_at_5,
                "recall_at_5": recall_at_5,
                "hit_rate": hit_rate,
                "recommended_cost": recommended_cost,
                "remaining_budget": budget - recommended_cost
            })

    summary_results.append({
        "budget": budget,
        "evaluation_cases": total_cases,
        "precision_at_5": np.mean(all_precision),
        "recall_at_5": np.mean(all_recall),
        "hit_rate": np.mean(all_hits),
        "recommendation_coverage": (
            cases_with_recommendations / total_cases
        ),
        "budget_compliance": (
            1 - budget_violations / total_cases
        ),
        "average_recommended_spend": total_spent / total_cases
    })


# ============================================================
# 7. SAVE EVALUATION REPORTS
# ============================================================

case_report = pd.DataFrame(case_results)
summary_report = pd.DataFrame(summary_results)

case_report.to_csv(
    REPORT_DIR / "train_test_case_results.csv",
    index=False
)

summary_report.to_csv(
    REPORT_DIR / "train_test_recommendation_metrics.csv",
    index=False
)


# ============================================================
# 8. GENERATE EVALUATION CHART
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    summary_report["budget"],
    summary_report["precision_at_5"],
    marker="o",
    label="Precision@5"
)

plt.plot(
    summary_report["budget"],
    summary_report["recall_at_5"],
    marker="s",
    label="Recall@5"
)

plt.plot(
    summary_report["budget"],
    summary_report["hit_rate"],
    marker="^",
    label="Hit Rate"
)

plt.plot(
    summary_report["budget"],
    summary_report["recommendation_coverage"],
    marker="d",
    label="Recommendation Coverage"
)

plt.xlabel("Additional Budget (Dataset Price Units)")
plt.ylabel("Evaluation Score")
plt.title("CartWise AI - Recommendation Performance by Budget")

plt.xticks(BUDGET_LEVELS)
plt.ylim(0, 1.05)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

chart_path = (
    VISUALIZATION_DIR / "train_test_recommendation_metrics.png"
)

plt.savefig(chart_path, dpi=300)
plt.close()


# ============================================================
# 9. DISPLAY FINAL RESULTS
# ============================================================

print("\nRECOMMENDATION EVALUATION")
print("-" * 65)

print(
    summary_report.round(4).to_string(index=False)
)

print("\nOUTPUT FILES")
print("-" * 40)
print(REPORT_DIR / "train_test_rule_evaluation.csv")
print(REPORT_DIR / "train_test_case_results.csv")
print(REPORT_DIR / "train_test_recommendation_metrics.csv")
print(chart_path)

print("\nWEEK 4 TRAIN-TEST EVALUATION COMPLETED")
print("=" * 65)
