
import json
import time

import pandas as pd

from week5_utils import (
    RANDOM_STATE,
    BUDGET_LEVELS,
    MAX_RECOMMENDATIONS,
    create_output_directories,
    load_data,
    create_grouped_split,
    generate_rules,
    evaluate_configuration,
    REPORT_DIR,
)


# ============================================================
# CARTWISE AI - WEEK 5 FINAL MODEL VALIDATION
# ============================================================

# Best configuration selected using Grid Search validation results.
BEST_MIN_SUPPORT = 0.005
BEST_MIN_CONFIDENCE = 0.20
BEST_MAX_LEN = 2

TEST_RESULTS_FILE = REPORT_DIR / "optimized_test_results.csv"
TEST_BUDGET_FILE = REPORT_DIR / "optimized_test_budget_results.csv"
VALIDATION_SUMMARY_FILE = REPORT_DIR / "final_model_validation.json"


def main():

    create_output_directories()

    print("\n" + "=" * 65)
    print("CARTWISE AI - FINAL OPTIMIZED MODEL VALIDATION")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    basket, prices, price_map = load_data()

    print(f"\nTotal transactions: {len(basket)}")
    print(f"Number of products: {basket.shape[1]}")

    # --------------------------------------------------------
    # 2. RECREATE THE SAME GROUPED SPLIT
    # --------------------------------------------------------

    train_data, validation_data, test_data = create_grouped_split(
        basket
    )

    print("\nGrouped data split:")
    print(f"Training transactions: {len(train_data)}")
    print(f"Validation transactions: {len(validation_data)}")
    print(f"Test transactions: {len(test_data)}")

    # --------------------------------------------------------
    # 3. TRAIN USING TRAINING DATA ONLY
    # --------------------------------------------------------

    print("\nSelected Grid Search configuration:")
    print(f"Minimum support: {BEST_MIN_SUPPORT}")
    print(f"Minimum confidence: {BEST_MIN_CONFIDENCE}")
    print(f"Maximum itemset length: {BEST_MAX_LEN}")

    start_time = time.time()

    frequent_itemsets, rules, mining_time = generate_rules(
        train_data,
        min_support=BEST_MIN_SUPPORT,
        min_confidence=BEST_MIN_CONFIDENCE,
        max_len=BEST_MAX_LEN
    )

    print(f"\nFrequent itemsets: {len(frequent_itemsets)}")
    print(f"Association rules: {len(rules)}")
    print(f"Mining time: {mining_time:.4f} seconds")

    # --------------------------------------------------------
    # 4. FINAL EVALUATION ON UNTOUCHED TEST DATA
    # --------------------------------------------------------

    print("\nEvaluating on the untouched test dataset...")

    test_summary, test_budget_results = evaluate_configuration(
        test_data,
        rules,
        price_map,
        budgets=BUDGET_LEVELS,
        max_recommendations=MAX_RECOMMENDATIONS
    )

    total_time = time.time() - start_time

    # --------------------------------------------------------
    # 5. SAVE TEST METRICS
    # --------------------------------------------------------

    test_summary_output = {
        "project": "CartWise AI",
        "method": "Grid Search",
        "random_state": RANDOM_STATE,
        "min_support": BEST_MIN_SUPPORT,
        "min_confidence": BEST_MIN_CONFIDENCE,
        "max_len": BEST_MAX_LEN,
        "frequent_itemsets": len(frequent_itemsets),
        "rule_count": len(rules),
        "mining_time_seconds": mining_time,
        "total_validation_execution_seconds": total_time,
        "test_set_used": True,
        "test_metrics": test_summary
    }

    with open(
        VALIDATION_SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            test_summary_output,
            file,
            indent=4
        )

    pd.DataFrame([test_summary_output]).to_csv(
        TEST_RESULTS_FILE,
        index=False
    )

    test_budget_results.to_csv(
        TEST_BUDGET_FILE,
        index=False
    )

    # --------------------------------------------------------
    # 6. DISPLAY FINAL RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL TEST SET RESULTS")
    print("=" * 65)

    for metric, value in test_summary.items():

        if isinstance(value, (int, float)):
            print(f"{metric}: {value:.6f}")
        else:
            print(f"{metric}: {value}")

    print("\n" + "=" * 65)
    print("FINAL VALIDATION COMPLETED")
    print("=" * 65)

    print("\nSaved files:")
    print(TEST_RESULTS_FILE)
    print(TEST_BUDGET_FILE)
    print(VALIDATION_SUMMARY_FILE)

    print(
        "\nThe model configuration was selected using validation data "
        "before this final test evaluation."
    )


if __name__ == "__main__":
    main()
