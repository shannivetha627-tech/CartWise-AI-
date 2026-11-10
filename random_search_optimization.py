
import itertools
import json
import random
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
# CARTWISE AI - WEEK 5 RANDOM SEARCH OPTIMIZATION
# ============================================================

MIN_SUPPORT_VALUES = [0.005, 0.01, 0.02, 0.03, 0.05]
MIN_CONFIDENCE_VALUES = [0.20, 0.30, 0.40, 0.50, 0.60]
MAX_LEN_VALUES = [2, 3, 4]

NUMBER_OF_TRIALS = 25

RESULT_FILE = REPORT_DIR / "random_search_results.csv"
BEST_FILE = REPORT_DIR / "random_search_best_configuration.json"


def main():

    create_output_directories()

    print("\n" + "=" * 65)
    print("CARTWISE AI - RANDOM SEARCH OPTIMIZATION")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    basket, prices, price_map = load_data()

    print(f"\nTotal transactions: {len(basket)}")
    print(f"Number of products: {basket.shape[1]}")

    # --------------------------------------------------------
    # 2. CREATE GROUPED DATA SPLIT
    # --------------------------------------------------------

    train_data, validation_data, test_data = create_grouped_split(
        basket
    )

    print("\nGrouped data split:")
    print(f"Training transactions: {len(train_data)}")
    print(f"Validation transactions: {len(validation_data)}")
    print(f"Test transactions: {len(test_data)}")

    # --------------------------------------------------------
    # 3. DEFINE PARAMETER SPACE
    # --------------------------------------------------------

    all_combinations = list(
        itertools.product(
            MIN_SUPPORT_VALUES,
            MIN_CONFIDENCE_VALUES,
            MAX_LEN_VALUES
        )
    )

    # Fixed random seed ensures reproducible selection.
    rng = random.Random(RANDOM_STATE)

    selected_combinations = rng.sample(
        all_combinations,
        NUMBER_OF_TRIALS
    )

    print("\nTotal possible configurations:", len(all_combinations))
    print("Random configurations selected:", len(selected_combinations))

    # --------------------------------------------------------
    # 4. RUN RANDOM SEARCH
    # --------------------------------------------------------

    results = []
    start_time = time.time()

    for trial_number, combination in enumerate(
        selected_combinations,
        start=1
    ):

        min_support, min_confidence, max_len = combination

        print("\n" + "-" * 65)
        print(f"Trial {trial_number}/{NUMBER_OF_TRIALS}")
        print(
            f"min_support={min_support}, "
            f"min_confidence={min_confidence}, "
            f"max_len={max_len}"
        )

        try:

            # Generate association rules using training data only.
            frequent_itemsets, rules, mining_time = generate_rules(
                train_data,
                min_support=min_support,
                min_confidence=min_confidence,
                max_len=max_len
            )

            # Evaluate using validation data.
            summary, budget_results = evaluate_configuration(
                validation_data,
                rules,
                price_map,
                budgets=BUDGET_LEVELS,
                max_recommendations=MAX_RECOMMENDATIONS
            )

            row = {
                "method": "Random Search",
                "trial": trial_number,
                "min_support": min_support,
                "min_confidence": min_confidence,
                "max_len": max_len,
                "frequent_itemsets": len(frequent_itemsets),
                "rule_count": len(rules),
                "mining_time_seconds": mining_time,
                "status": "Success",
                **summary,
                "test_set_used": False
            }

            print(
                f"Optimization score: "
                f"{summary.get('optimization_score', 0):.6f}"
            )

        except Exception as error:

            print("Trial failed:", str(error))

            row = {
                "method": "Random Search",
                "trial": trial_number,
                "min_support": min_support,
                "min_confidence": min_confidence,
                "max_len": max_len,
                "status": "Failed",
                "error": str(error),
                "test_set_used": False
            }

        results.append(row)

        # Save progress after every trial.
        pd.DataFrame(results).to_csv(
            RESULT_FILE,
            index=False
        )

    # --------------------------------------------------------
    # 5. IDENTIFY BEST CONFIGURATION
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    successful_results = results_df[
        (results_df["status"] == "Success")
        & results_df["optimization_score"].notna()
    ].copy()

    if successful_results.empty:
        raise RuntimeError(
            "No Random Search configuration completed successfully."
        )

    successful_results = successful_results.sort_values(
        by="optimization_score",
        ascending=False
    )

    best_configuration = successful_results.iloc[0].to_dict()

    # --------------------------------------------------------
    # 6. SAVE BEST CONFIGURATION
    # --------------------------------------------------------

    best_output = {
        "method": "Random Search",
        "random_state": RANDOM_STATE,
        "number_of_trials": NUMBER_OF_TRIALS,
        "total_possible_configurations": len(all_combinations),
        "successful_trials": len(successful_results),
        "failed_trials": NUMBER_OF_TRIALS - len(successful_results),
        "best_configuration": {
            key: (
                value.item()
                if hasattr(value, "item")
                else value
            )
            for key, value in best_configuration.items()
        },
        "test_set_used": False
    }

    with open(BEST_FILE, "w", encoding="utf-8") as file:
        json.dump(best_output, file, indent=4)

    # --------------------------------------------------------
    # 7. DISPLAY RESULTS
    # --------------------------------------------------------

    total_time = time.time() - start_time

    print("\n" + "=" * 65)
    print("RANDOM SEARCH COMPLETED")
    print("=" * 65)

    print(f"\nSuccessful trials: {len(successful_results)}")
    print(
        f"Failed trials: "
        f"{NUMBER_OF_TRIALS - len(successful_results)}"
    )
    print(f"Total execution time: {total_time:.2f} seconds")

    print("\nBEST CONFIGURATION:")
    print(
        successful_results[
            [
                "min_support",
                "min_confidence",
                "max_len",
                "optimization_score",
                "precision_at_5",
                "recall_at_5",
                "hit_rate",
                "coverage",
                "rule_count"
            ]
        ].head(1).to_string(index=False)
    )

    print("\nTOP 5 CONFIGURATIONS:")

    print(
        successful_results[
            [
                "min_support",
                "min_confidence",
                "max_len",
                "optimization_score",
                "precision_at_5",
                "recall_at_5",
                "hit_rate",
                "coverage"
            ]
        ].head(5).to_string(index=False)
    )

    print("\nSaved files:")
    print(RESULT_FILE)
    print(BEST_FILE)

    print("\nThe test dataset was not used during optimization.")


if __name__ == "__main__":
    main()
