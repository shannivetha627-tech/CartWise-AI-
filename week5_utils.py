
from pathlib import Path
import inspect
import time

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# CARTWISE AI - WEEK 5 SHARED UTILITIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

BASKET_FILE = BASE_DIR / "data" / "processed" / "basket_matrix.csv"
PRICE_FILE = BASE_DIR / "data" / "processed" / "product_price_lookup.csv"

REPORT_DIR = BASE_DIR / "data" / "reports" / "week5"
VISUALIZATION_DIR = BASE_DIR / "visualizations" / "week5"

RANDOM_STATE = 42
BUDGET_LEVELS = [2.00, 5.00, 10.00, 15.00]
MAX_RECOMMENDATIONS = 5

RULE_COLUMNS = [
    "antecedents",
    "consequents",
    "support",
    "confidence",
    "lift",
]


# ============================================================
# 1. CREATE OUTPUT DIRECTORIES
# ============================================================

def create_output_directories():
    """Create folders for Week 5 outputs."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATASETS
# ============================================================

def load_data():
    """Load the processed basket matrix and product prices."""

    if not BASKET_FILE.exists():
        raise FileNotFoundError(
            f"Basket dataset not found: {BASKET_FILE}"
        )

    if not PRICE_FILE.exists():
        raise FileNotFoundError(
            f"Price lookup not found: {PRICE_FILE}"
        )

    basket = (
        pd.read_csv(BASKET_FILE, index_col=0)
        .fillna(False)
        .astype(bool)
    )

    prices = pd.read_csv(PRICE_FILE)

    required_price_columns = {
        "StockCode",
        "UnitPrice"
    }

    if not required_price_columns.issubset(prices.columns):
        raise ValueError(
            f"Price file must contain: {required_price_columns}"
        )

    if basket.empty or prices.empty:
        raise ValueError("The input datasets are empty.")

    if prices["StockCode"].duplicated().any():
        raise ValueError(
            "Duplicate product IDs found in price lookup."
        )

    prices["UnitPrice"] = pd.to_numeric(
        prices["UnitPrice"],
        errors="coerce"
    )

    if prices["UnitPrice"].isna().any():
        raise ValueError("Invalid product prices found.")

    if (prices["UnitPrice"] < 0).any():
        raise ValueError("Negative product prices found.")

    price_map = (
        prices.set_index("StockCode")["UnitPrice"].to_dict()
    )

    missing_products = set(basket.columns) - set(price_map)

    if missing_products:
        raise ValueError(
            f"Missing prices for products: {sorted(missing_products)}"
        )

    return basket, prices, price_map


# ============================================================
# 3. GROUPED TRAIN-VALIDATION-TEST SPLIT
# ============================================================

def create_grouped_split(basket):
    """
    Create reproducible grouped train-validation-test partitions.

    Approximate proportions:
        Training:   64%
        Validation: 16%
        Testing:    20%

    Identical basket patterns are kept in the same partition
    to reduce data leakage caused by repeated transactions.
    """

    # Create a unique signature for every basket pattern.
    signatures = (
        basket.astype(np.uint8)
        .astype(str)
        .agg("".join, axis=1)
        .to_numpy()
    )

    # --------------------------------------------------------
    # First split: reserve 20% of unique groups for testing.
    # --------------------------------------------------------

    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    train_val_positions, test_positions = next(
        first_split.split(
            basket,
            groups=signatures
        )
    )

    train_val = basket.iloc[train_val_positions]

    train_val_groups = signatures[train_val_positions]

    # --------------------------------------------------------
    # Second split: reserve 20% of remaining groups for
    # validation.
    # --------------------------------------------------------

    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    train_positions, validation_positions = next(
        second_split.split(
            train_val,
            groups=train_val_groups
        )
    )

    train_data = train_val.iloc[train_positions]

    validation_data = train_val.iloc[validation_positions]

    test_data = basket.iloc[test_positions]

    # --------------------------------------------------------
    # Verify that identical basket patterns do not overlap.
    #
    # IMPORTANT:
    # Use positional indices here, not DataFrame index labels.
    # This fixes the previous IndexError.
    # --------------------------------------------------------

    train_groups = set(
        signatures[train_val_positions[train_positions]]
    )

    validation_groups = set(
        signatures[train_val_positions[validation_positions]]
    )

    test_groups = set(
        signatures[test_positions]
    )

    assert train_groups.isdisjoint(validation_groups), (
        "Data leakage detected between training and validation."
    )

    assert train_groups.isdisjoint(test_groups), (
        "Data leakage detected between training and testing."
    )

    assert validation_groups.isdisjoint(test_groups), (
        "Data leakage detected between validation and testing."
    )

    if train_data.empty or validation_data.empty or test_data.empty:
        raise ValueError(
            "One of the data partitions is empty."
        )

    return train_data, validation_data, test_data


# ============================================================
# 4. GENERATE APRIORI ASSOCIATION RULES
# ============================================================

def generate_rules(
    train_data,
    min_support,
    min_confidence,
    max_len
):
    """Generate frequent itemsets and association rules."""

    start_time = time.perf_counter()

    frequent_itemsets = apriori(
        train_data,
        min_support=float(min_support),
        use_colnames=True,
        max_len=int(max_len),
        low_memory=True
    )

    if frequent_itemsets.empty:

        rules = pd.DataFrame(columns=RULE_COLUMNS)

    else:

        association_parameters = inspect.signature(
            association_rules
        ).parameters

        extra_arguments = {}

        # Compatibility with newer mlxtend versions.
        if "num_itemsets" in association_parameters:
            extra_arguments["num_itemsets"] = len(train_data)

        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=float(min_confidence),
            **extra_arguments
        )

        if not rules.empty:

            rules = rules.sort_values(
                ["confidence", "lift", "support"],
                ascending=False
            ).reset_index(drop=True)

    elapsed_time = time.perf_counter() - start_time

    return frequent_itemsets, rules, elapsed_time


# ============================================================
# 5. PRODUCT RECOMMENDATION ENGINE
# ============================================================

def recommend_products(
    selected_products,
    rules,
    price_map,
    budget,
    max_recommendations=MAX_RECOMMENDATIONS
):
    """
    Generate recommendations using confidence, lift and support.

    Products are filtered according to the available budget.
    """

    selected_products = set(selected_products)

    if rules.empty:
        return []

    product_scores = {}

    for rule in rules.itertuples(index=False):

        antecedents = set(rule.antecedents)

        consequents = set(rule.consequents)

        if not antecedents.issubset(selected_products):
            continue

        for product in consequents:

            if product in selected_products:
                continue

            score = (
                float(rule.confidence),
                float(rule.lift),
                float(rule.support)
            )

            if (
                product not in product_scores
                or score > product_scores[product]
            ):
                product_scores[product] = score

    # Rank recommendations by confidence, lift and support.
    ranked_products = sorted(
        product_scores.items(),
        key=lambda item: (
            -item[1][0],
            -item[1][1],
            -item[1][2],
            float(price_map[item[0]]),
            str(item[0])
        )
    )

    recommendations = []

    remaining_budget = float(budget)

    for product, score in ranked_products:

        product_price = float(price_map[product])

        if product_price <= remaining_budget:

            recommendations.append(product)

            remaining_budget -= product_price

        if len(recommendations) >= max_recommendations:
            break

    return recommendations


# ============================================================
# 6. EVALUATE MODEL CONFIGURATION
# ============================================================

def evaluate_configuration(
    evaluation_data,
    rules,
    price_map,
    budgets=None,
    max_recommendations=MAX_RECOMMENDATIONS
):
    """
    Evaluate recommendations for every product in each transaction.

    Each product is treated as a simulated selected product,
    following the evaluation approach used in Week 4.
    """

    if budgets is None:
        budgets = BUDGET_LEVELS

    start_time = time.perf_counter()

    budget_results = []

    all_products = set(evaluation_data.columns)

    mean_lift = (
        float(rules["lift"].mean())
        if not rules.empty
        else 0.0
    )

    # --------------------------------------------------------
    # Evaluate each budget level.
    # --------------------------------------------------------

    for budget in budgets:

        precision_values = []

        recall_values = []

        hit_values = []

        covered_cases = 0

        total_cases = 0

        total_spend = 0.0

        budget_violations = 0

        recommended_products = set()

        # Iterate through transactions.
        for row in evaluation_data.itertuples(
            index=False,
            name=None
        ):

            actual_products = {
                product
                for product, present in zip(
                    evaluation_data.columns,
                    row
                )
                if bool(present)
            }

            # Simulate selecting each product individually.
            for selected_product in actual_products:

                relevant_products = (
                    actual_products - {selected_product}
                )

                recommendations = recommend_products(
                    selected_products={selected_product},
                    rules=rules,
                    price_map=price_map,
                    budget=budget,
                    max_recommendations=max_recommendations
                )

                recommended_products.update(recommendations)

                hits = (
                    set(recommendations) & relevant_products
                )

                total_cases += 1

                # Precision@K.
                precision_values.append(
                    len(hits) / max_recommendations
                )

                # Recall@K.
                if relevant_products:

                    recall_values.append(
                        len(hits) / len(relevant_products)
                    )

                # Hit Rate.
                hit_values.append(
                    1.0 if hits else 0.0
                )

                # Recommendation coverage.
                if recommendations:
                    covered_cases += 1

                # Calculate recommendation spending.
                spend = sum(
                    float(price_map[product])
                    for product in recommendations
                )

                total_spend += spend

                if spend > budget + 1e-9:
                    budget_violations += 1

        # ----------------------------------------------------
        # Store metrics for this budget.
        # ----------------------------------------------------

        budget_results.append({

            "budget": float(budget),

            "precision_at_5": (
                float(np.mean(precision_values))
                if precision_values else 0.0
            ),

            "recall_at_5": (
                float(np.mean(recall_values))
                if recall_values else 0.0
            ),

            "hit_rate": (
                float(np.mean(hit_values))
                if hit_values else 0.0
            ),

            "coverage": (
                covered_cases / total_cases
                if total_cases else 0.0
            ),

            "budget_compliance": (
                1.0 - budget_violations / total_cases
                if total_cases else 1.0
            ),

            "average_spend": (
                total_spend / total_cases
                if total_cases else 0.0
            ),

            "product_coverage": (
                len(recommended_products) / len(all_products)
                if all_products else 0.0
            ),

            "evaluation_cases": total_cases,

            "rule_count": len(rules),

            "mean_lift": mean_lift

        })

    elapsed_time = time.perf_counter() - start_time

    budget_frame = pd.DataFrame(budget_results)

    # --------------------------------------------------------
    # Calculate overall average metrics across budgets.
    # --------------------------------------------------------

    summary = {

        "precision_at_5": float(
            budget_frame["precision_at_5"].mean()
        ),

        "recall_at_5": float(
            budget_frame["recall_at_5"].mean()
        ),

        "hit_rate": float(
            budget_frame["hit_rate"].mean()
        ),

        "coverage": float(
            budget_frame["coverage"].mean()
        ),

        "budget_compliance": float(
            budget_frame["budget_compliance"].mean()
        ),

        "average_spend": float(
            budget_frame["average_spend"].mean()
        ),

        "product_coverage": float(
            budget_frame["product_coverage"].mean()
        ),

        "rule_count": len(rules),

        "mean_lift": mean_lift,

        "evaluation_time_seconds": elapsed_time

    }

    # --------------------------------------------------------
    # Weighted optimization objective.
    #
    # Precision@5: 40%
    # Recall@5:    30%
    # Hit Rate:    20%
    # Coverage:    10%
    # --------------------------------------------------------

    summary["optimization_score"] = (

        0.40 * summary["precision_at_5"]

        + 0.30 * summary["recall_at_5"]

        + 0.20 * summary["hit_rate"]

        + 0.10 * summary["coverage"]

    )

    return summary, budget_frame


# ============================================================
# END OF WEEK 5 SHARED UTILITIES
# ============================================================

