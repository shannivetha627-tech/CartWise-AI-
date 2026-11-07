
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mlxtend.frequent_patterns import apriori, association_rules


# ==========================================================
# CARTWISE AI - WEEK 4: ERROR ANALYSIS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "processed" / "basket_matrix.csv"
REPORT_DIR = BASE_DIR / "data" / "reports"
VISUAL_DIR = BASE_DIR / "visualizations"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
VISUAL_DIR.mkdir(parents=True, exist_ok=True)


# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.20
TOP_K = 5

MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.30
MAX_LEN = 3


# ----------------------------------------------------------
# 1. LOAD DATASET
# ----------------------------------------------------------

def load_data():

    print("\nLoading basket matrix...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    data = pd.read_csv(DATA_FILE, index_col=0)
    data = data.astype(bool)

    print("Transactions:", len(data))
    print("Products:", len(data.columns))

    return data


# ----------------------------------------------------------
# 2. TRAIN-TEST SPLIT
# ----------------------------------------------------------

def split_data(data):

    rng = np.random.default_rng(RANDOM_STATE)

    indices = rng.permutation(len(data))

    test_count = int(round(len(data) * TEST_SIZE))

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    train_data = (
        data.iloc[train_indices]
        .reset_index(drop=True)
    )

    test_data = (
        data.iloc[test_indices]
        .reset_index(drop=True)
    )

    print("\nTRAIN-TEST SPLIT")

    print("Training transactions:", len(train_data))
    print("Testing transactions:", len(test_data))

    return train_data, test_data


# ----------------------------------------------------------
# 3. TRAIN ASSOCIATION RULE MODEL
# ----------------------------------------------------------

def train_rules(train_data):

    print("\nMining association rules using training data...")

    frequent_itemsets = apriori(
        train_data,
        min_support=MIN_SUPPORT,
        use_colnames=True,
        max_len=MAX_LEN
    )

    if frequent_itemsets.empty:

        raise ValueError(
            "No frequent itemsets found. "
            "Check the minimum support threshold."
        )

    try:

        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=MIN_CONFIDENCE,
            num_itemsets=len(train_data)
        )

    except TypeError:

        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=MIN_CONFIDENCE
        )

    print("Frequent itemsets:", len(frequent_itemsets))
    print("Association rules:", len(rules))

    return rules


# ----------------------------------------------------------
# 4. GENERATE RECOMMENDATIONS
# ----------------------------------------------------------

def get_recommendations(selected_product, actual_basket, rules):

    candidates = {}

    for _, rule in rules.iterrows():

        antecedent = set(rule["antecedents"])
        consequent = set(rule["consequents"])

        # Check whether the selected product
        # belongs to the rule antecedent.

        if selected_product not in antecedent:
            continue

        for product in consequent:

            # Exclude only the product already selected.
            # Do not exclude actual basket products.
            # They are required to evaluate correct predictions.

            if product == selected_product:
                continue

            score = (
                float(rule["confidence"])
                * float(rule["lift"])
            )

            if (
                product not in candidates
                or score > candidates[product]
            ):

                candidates[product] = score

    ranked = sorted(
        candidates.items(),
        key=lambda item: item[1],
        reverse=True
    )

    recommendations = [
        product
        for product, score in ranked[:TOP_K]
    ]

    return recommendations


# ----------------------------------------------------------
# 5. ERROR ANALYSIS
# ----------------------------------------------------------

def analyse_errors(test_data, rules):

    print("\nANALYSING RECOMMENDATION ERRORS...")

    case_results = []

    product_errors = {}

    total_tp = 0
    total_fp = 0
    total_fn = 0

    evaluated_transactions = 0
    transactions_with_recommendations = 0

    for transaction_index, transaction in test_data.iterrows():

        actual_basket = set(
            transaction[transaction].index.tolist()
        )

        if len(actual_basket) < 2:
            continue

        evaluated_transactions += 1

        # Use one product as the customer's selected product.
        selected_product = sorted(actual_basket)[0]

        # Remaining products are treated as relevant products.
        relevant_products = (
            actual_basket - {selected_product}
        )

        recommendations = get_recommendations(
            selected_product,
            actual_basket,
            rules
        )

        if recommendations:
            transactions_with_recommendations += 1

        recommended_set = set(recommendations)

        # Classification of recommendation outcomes.

        true_positives = (
            recommended_set & relevant_products
        )

        false_positives = (
            recommended_set - relevant_products
        )

        false_negatives = (
            relevant_products - recommended_set
        )

        tp = len(true_positives)
        fp = len(false_positives)
        fn = len(false_negatives)

        # Precision@5 uses fixed K=5.
        # Empty recommendation positions count as non-hits.

        precision = tp / TOP_K

        recall = (
            tp / len(relevant_products)
            if relevant_products else 0
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Track product-level false positives.

        for product in false_positives:

            if product not in product_errors:

                product_errors[product] = {
                    "false_positive_count": 0,
                    "false_negative_count": 0
                }

            product_errors[product][
                "false_positive_count"
            ] += 1

        # Track product-level false negatives.

        for product in false_negatives:

            if product not in product_errors:

                product_errors[product] = {
                    "false_positive_count": 0,
                    "false_negative_count": 0
                }

            product_errors[product][
                "false_negative_count"
            ] += 1

        case_results.append({

            "transaction_index": int(transaction_index),

            "selected_product": selected_product,

            "actual_relevant_products": ", ".join(
                sorted(relevant_products)
            ),

            "recommended_products": ", ".join(
                recommendations
            ),

            "true_positives": ", ".join(
                sorted(true_positives)
            ),

            "false_positives": ", ".join(
                sorted(false_positives)
            ),

            "false_negatives": ", ".join(
                sorted(false_negatives)
            ),

            "TP": tp,
            "FP": fp,
            "FN": fn,

            "precision_at_5": precision,
            "recall_at_5": recall

        })

    results = pd.DataFrame(case_results)

    if results.empty:

        raise ValueError(
            "No eligible test transactions were available."
        )

    # ------------------------------------------------------
    # OVERALL METRICS
    # ------------------------------------------------------

    overall_precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp else 0
    )

    overall_recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn else 0
    )

    f1_score = (
        2 * overall_precision * overall_recall
        / (overall_precision + overall_recall)
        if overall_precision + overall_recall else 0
    )

    summary = pd.DataFrame([{

        "evaluated_transactions": evaluated_transactions,

        "transactions_with_recommendations":
            transactions_with_recommendations,

        "top_k": TOP_K,

        "true_positives": total_tp,

        "false_positives": total_fp,

        "false_negatives": total_fn,

        "micro_precision": overall_precision,

        "micro_recall": overall_recall,

        "micro_f1_score": f1_score,

        "mean_case_precision_at_5":
            results["precision_at_5"].mean(),

        "mean_case_recall_at_5":
            results["recall_at_5"].mean()

    }])

    # ------------------------------------------------------
    # PRODUCT-LEVEL ERROR SUMMARY
    # ------------------------------------------------------

    product_summary = pd.DataFrame([

        {
            "product": product,
            **counts
        }

        for product, counts in product_errors.items()

    ])

    if not product_summary.empty:

        product_summary = product_summary.sort_values(
            [
                "false_negative_count",
                "false_positive_count"
            ],
            ascending=False
        )

    return results, summary, product_summary


# ----------------------------------------------------------
# 6. SAVE REPORTS AND VISUALIZATIONS
# ----------------------------------------------------------

def save_outputs(results, summary, product_summary):

    # Save transaction-level results.

    results.to_csv(
        REPORT_DIR / "error_analysis_case_results.csv",
        index=False
    )

    # Save overall evaluation metrics.

    summary.to_csv(
        REPORT_DIR / "error_analysis_summary.csv",
        index=False
    )

    # Save product-level errors.

    product_summary.to_csv(
        REPORT_DIR / "product_error_analysis.csv",
        index=False
    )

    print("\nERROR ANALYSIS SUMMARY")

    print(
        summary.to_string(index=False)
    )

    print("\nMOST FREQUENT PRODUCT ERRORS")

    if product_summary.empty:

        print("No product-level errors recorded.")

    else:

        print(
            product_summary.head(10).to_string(index=False)
        )

    # ------------------------------------------------------
    # ERROR COUNT VISUALIZATION
    # ------------------------------------------------------

    counts = summary.iloc[0]

    labels = [
        "True Positives",
        "False Positives",
        "False Negatives"
    ]

    values = [
        counts["true_positives"],
        counts["false_positives"],
        counts["false_negatives"]
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(labels, values)

    plt.title(
        "CartWise AI - Recommendation Error Analysis"
    )

    plt.ylabel("Number of Products")

    plt.tight_layout()

    plt.savefig(
        VISUAL_DIR / "error_analysis.png",
        dpi=300
    )

    plt.close()

    print("\nFILES SAVED")

    print(
        "data/reports/error_analysis_case_results.csv"
    )

    print(
        "data/reports/error_analysis_summary.csv"
    )

    print(
        "data/reports/product_error_analysis.csv"
    )

    print(
        "visualizations/error_analysis.png"
    )


# ----------------------------------------------------------
# 7. MAIN EXECUTION
# ----------------------------------------------------------

def main():

    print("=" * 55)

    print(
        "CARTWISE AI - WEEK 4 ERROR ANALYSIS"
    )

    print("=" * 55)

    data = load_data()

    train_data, test_data = split_data(data)

    rules = train_rules(train_data)

    results, summary, product_summary = analyse_errors(
        test_data,
        rules
    )

    save_outputs(
        results,
        summary,
        product_summary
    )

    print(
        "\nERROR ANALYSIS COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":

    main()
