
import itertools
import time
import json

import pandas as pd

from week5_utils import (
    create_output_directories,
    load_data,
    create_grouped_split,
    generate_rules,
    evaluate_configuration,
    REPORT_DIR,
    RANDOM_STATE,
    BUDGET_LEVELS,
    MAX_RECOMMENDATIONS
)


# ============================================================
# CARTWISE AI - WEEK 5 GRID SEARCH OPTIMIZATION
# ============================================================

MIN_SUPPORT_VALUES = [0.005, 0.01, 0.02, 0.03, 0.05]
MIN_CONFIDENCE_VALUES = [0.20, 0.30, 0.40, 0.50, 0.60]
MAX_LEN_VALUES = [2, 3, 4]

OUTPUT_FILE = REPORT_DIR / "grid_search_results.csv"
BEST_CONFIG_FILE = REPORT_DIR / "grid_search_best_configuration.json"


def main():

    print("\n" + "=" * 65)
    print("CARTWISE AI - WEEK 5 GRID SEARCH OPTIMIZATION")
    print("=" * 65)

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    print("\n[1/5] Loading datasets...")

    create_output_directories()

    # load_data() returns three values.
    basket, prices, price_map = load_data()

    print(f"Transactions: {len(basket)}")
    print(f"Products: {basket.shape[1]}")
    print(f"Price records: {len(prices)}")

    # --------------------------------------------------------
    # 2. CREATE GROUPED DATA SPLIT
    # --------------------------------------------------------

    print("\n[2/5] Creating grouped data split...")

    train_data, validation_data, test_data = create_grouped_split(
        basket
    )

    print(f"Training transactions: {len(train_data)}")
    print(f"Validation transactions: {len(validation_data)}")
    print(f"Test transactions: {len(test_data)}")

    print("Test data is reserved for final validation.")

    # --------------------------------------------------------
    # 3. GENERATE PARAMETER COMBINATIONS
    # --------------------------------------------------------

    print("\n[3/5] Preparing Grid Search combinations...")

    combinations = list(
        itertools.product(
            MIN_SUPPORT_VALUES,
            MIN_CONFIDENCE_VALUES,
            MAX_LEN_VALUES
        )
    )

    print(f"Min support values: {MIN_SUPPORT_VALUES}")
    print(f"Min confidence values: {MIN_CONFIDENCE_VALUES}")
    print(f"Maximum rule lengths: {MAX_LEN_VALUES}")
    print(f"Total configurations: {len(combinations)}")

    # --------------------------------------------------------
    # 4. EVALUATE EACH CONFIGURATION
    # --------------------------------------------------------

    print("\n[4/5] Evaluating configurations...")

    results = []

    for index, (support, confidence, max_len) in enumerate(
        combinations,
        start=1
    ):

        print("\n" + "-" * 55)
        print(f"Configuration {index}/{len(combinations)}")

        print(
            f"min_support={support}, "
            f"min_confidence={confidence}, "
            f"max_len={max_len}"
        )

        try:

            # Generate frequent itemsets and association rules.
            frequent_itemsets, rules, mining_time = generate_rules(
                train_data,
                support,
                confidence,
                max_len
            )

            # Evaluate recommendations using validation data.
            metrics, budget_results = evaluate_configuration(
                validation_data,
                rules,
                price_map,
                budgets=BUDGET_LEVELS,
                max_recommendations=MAX_RECOMMENDATIONS
            )

            result = {
                "configuration": index,
                "min_support": support,
                "min_confidence": confidence,
                "max_len": max_len,
                "frequent_itemsets": len(frequent_itemsets),
                "rule_count": len(rules),
                "mining_time_seconds": mining_time,
                **metrics
            }

            results.append(result)

            print(f"Frequent itemsets: {len(frequent_itemsets)}")
            print(f"Association rules: {len(rules)}")
            print(f"Mining time: {mining_time:.4f} seconds")

            print(
                f"Precision@5: {metrics['precision_at_5']:.6f}"
            )

            print(
                f"Recall@5: {metrics['recall_at_5']:.6f}"
            )

            print(
                f"Optimization Score: "
                f"{metrics['optimization_score']:.6f}"
            )

        except Exception as error:

            print(f"Configuration failed: {error}")

            results.append({
                "configuration": index,
                "min_support": support,
                "min_confidence": confidence,
                "max_len": max_len,
                "error": str(error),
                "optimization_score": float("nan")
            })

        # Save progress after each configuration.
        # This preserves completed results if execution is interrupted.
        pd.DataFrame(results).to_csv(
            OUTPUT_FILE,
            index=False
        )

    # --------------------------------------------------------
    # 5. SAVE AND DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n[5/5] Saving Grid Search results...")

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    successful_results = results_df.dropna(
        subset=["optimization_score"]
    )

    if successful_results.empty:

        raise RuntimeError(
            "All Grid Search configurations failed. "
            "Check the error column in the results CSV."
        )

    successful_results = successful_results.sort_values(
        by="optimization_score",
        ascending=False
    ).reset_index(drop=True)

    best = successful_results.iloc[0]

    best_configuration = {
        "method": "Grid Search",
        "random_state": RANDOM_STATE,
        "min_support": float(best["min_support"]),
        "min_confidence": float(best["min_confidence"]),
        "max_len": int(best["max_len"]),
        "optimization_score": float(best["optimization_score"]),
        "precision_at_5": float(best["precision_at_5"]),
        "recall_at_5": float(best["recall_at_5"]),
        "hit_rate": float(best["hit_rate"]),
        "coverage": float(best["coverage"]),
        "budget_compliance": float(best["budget_compliance"]),
        "rule_count": int(best["rule_count"]),
        "test_set_used": False
    }

    with open(
        BEST_CONFIG_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            best_configuration,
            file,
            indent=4
        )

    total_time = time.perf_counter() - start_time

    print("\n" + "=" * 65)
    print("GRID SEARCH RESULTS")
    print("=" * 65)

    print(
        f"Successful configurations: "
        f"{len(successful_results)}"
    )

    print(
        f"Failed configurations: "
        f"{len(results_df) - len(successful_results)}"
    )

    print("\nBEST CONFIGURATION")

    for key, value in best_configuration.items():
        print(f"{key}: {value}")

    print("\nTOP 5 CONFIGURATIONS")

    display_columns = [
        "min_support",
        "min_confidence",
        "max_len",
        "rule_count",
        "precision_at_5",
        "recall_at_5",
        "hit_rate",
        "coverage",
        "optimization_score"
    ]

    print(
        successful_results[display_columns]
        .head(5)
        .to_string(index=False)
    )

    print(
        f"\nTotal execution time: "
        f"{total_time:.2f} seconds"
    )

    print("\nFiles saved:")
    print(OUTPUT_FILE)
    print(BEST_CONFIG_FILE)

    print("\nGrid Search completed successfully.")


if __name__ == "__main__":
    main()

