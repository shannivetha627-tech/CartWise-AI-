
from pathlib import Path
import json
import time

import pandas as pd

from week5_utils import (
    BASE_DIR,
    REPORT_DIR,
    create_output_directories,
    load_data,
    create_grouped_split,
    generate_rules,
    evaluate_configuration,
)


# ============================================================
# CARTWISE AI - WEEK 5 BASELINE EVALUATION
# ============================================================

MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.30
MAX_LEN = 3


def main():

    print("\n" + "=" * 60)
    print("CARTWISE AI - WEEK 5 BASELINE EVALUATION")
    print("=" * 60)

    create_output_directories()

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    print("\n[1/5] Loading datasets...")

    basket, prices, price_map = load_data()

    print(f"Transactions: {len(basket)}")
    print(f"Products: {len(basket.columns)}")

    # --------------------------------------------------------
    # 2. Create grouped train-validation-test split
    # --------------------------------------------------------

    print("\n[2/5] Creating grouped data split...")

    train_data, validation_data, test_data = (
        create_grouped_split(basket)
    )

    print(f"Training transactions: {len(train_data)}")
    print(f"Validation transactions: {len(validation_data)}")
    print(f"Test transactions: {len(test_data)}")

    print("\nTest data is reserved for final validation.")

    # --------------------------------------------------------
    # 3. Generate baseline association rules
    # --------------------------------------------------------

    print("\n[3/5] Generating baseline Apriori rules...")

    frequent_itemsets, rules, mining_time = generate_rules(
        train_data=train_data,
        min_support=MIN_SUPPORT,
        min_confidence=MIN_CONFIDENCE,
        max_len=MAX_LEN,
    )

    print(f"Frequent itemsets: {len(frequent_itemsets)}")
    print(f"Association rules: {len(rules)}")
    print(f"Mining time: {mining_time:.4f} seconds")

    # --------------------------------------------------------
    # 4. Evaluate on validation data
    # --------------------------------------------------------

    print("\n[4/5] Evaluating baseline on validation data...")

    evaluation_start = time.perf_counter()

    summary, budget_results = evaluate_configuration(
        evaluation_data=validation_data,
        rules=rules,
        price_map=price_map,
    )

    evaluation_time = time.perf_counter() - evaluation_start

    # --------------------------------------------------------
    # 5. Save results
    # --------------------------------------------------------

    print("\n[5/5] Saving baseline results...")

    baseline_result = {
        "experiment": "Baseline",
        "min_support": MIN_SUPPORT,
        "min_confidence": MIN_CONFIDENCE,
        "max_len": MAX_LEN,
        "training_transactions": len(train_data),
        "validation_transactions": len(validation_data),
        "reserved_test_transactions": len(test_data),
        "frequent_itemsets": len(frequent_itemsets),
        "rule_count": len(rules),
        "mining_time_seconds": mining_time,
        "evaluation_time_seconds": evaluation_time,
        **summary,
    }

    baseline_df = pd.DataFrame([baseline_result])

    baseline_file = REPORT_DIR / "baseline_results.csv"
    budget_file = REPORT_DIR / "baseline_budget_results.csv"

    baseline_df.to_csv(baseline_file, index=False)
    budget_results.to_csv(budget_file, index=False)

    # Save baseline configuration for reproducibility.
    configuration = {
        "experiment": "Week 5 Baseline",
        "random_state": 42,
        "min_support": MIN_SUPPORT,
        "min_confidence": MIN_CONFIDENCE,
        "max_len": MAX_LEN,
        "budget_levels": [2.00, 5.00, 10.00, 15.00],
        "max_recommendations": 5,
        "split_method": "Grouped 64:16:20 train-validation-test",
        "test_set_used": False,
    }

    config_file = REPORT_DIR / "baseline_configuration.json"

    with open(config_file, "w", encoding="utf-8") as file:
        json.dump(configuration, file, indent=4)

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("BASELINE EVALUATION RESULTS")
    print("=" * 60)

    for metric, value in summary.items():
        print(f"{metric}: {value:.6f}")

    print("\nFiles saved:")
    print(baseline_file)
    print(budget_file)
    print(config_file)

    print("\nBaseline evaluation completed successfully.")


if __name__ == "__main__":
    main()
