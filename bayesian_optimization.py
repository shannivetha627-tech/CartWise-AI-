
import json
import time

import optuna
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
# CARTWISE AI - WEEK 5 BAYESIAN OPTIMIZATION
# ============================================================

NUMBER_OF_TRIALS = 25

MIN_SUPPORT_VALUES = [0.005, 0.01, 0.02, 0.03, 0.05]
MIN_CONFIDENCE_VALUES = [0.20, 0.30, 0.40, 0.50, 0.60]
MAX_LEN_VALUES = [2, 3, 4]

RESULT_FILE = REPORT_DIR / "bayesian_optimization_results.csv"
BEST_FILE = REPORT_DIR / "bayesian_best_configuration.json"


def main():

    create_output_directories()

    print("\n" + "=" * 65)
    print("CARTWISE AI - BAYESIAN OPTIMIZATION")
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
    # 3. CREATE OPTUNA STUDY
    # --------------------------------------------------------

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="CartWise_AI_Bayesian_Optimization"
    )

    results = []
    start_time = time.time()

    # --------------------------------------------------------
    # 4. DEFINE OBJECTIVE FUNCTION
    # --------------------------------------------------------

    def objective(trial):

        min_support = trial.suggest_categorical(
            "min_support",
            MIN_SUPPORT_VALUES
        )

        min_confidence = trial.suggest_categorical(
            "min_confidence",
            MIN_CONFIDENCE_VALUES
        )

        max_len = trial.suggest_categorical(
            "max_len",
            MAX_LEN_VALUES
        )

        print("\n" + "-" * 65)
        print(f"Trial {trial.number + 1}/{NUMBER_OF_TRIALS}")
        print(f"min_support: {min_support}")
        print(f"min_confidence: {min_confidence}")
        print(f"max_len: {max_len}")

        try:

            # Mine association rules using training data.
            frequent_itemsets, rules, mining_time = generate_rules(
                train_data,
                min_support=min_support,
                min_confidence=min_confidence,
                max_len=max_len
            )

            # Evaluate using validation data only.
            summary, budget_results = evaluate_configuration(
                validation_data,
                rules,
                price_map,
                budgets=BUDGET_LEVELS,
                max_recommendations=MAX_RECOMMENDATIONS
            )

            score = float(summary["optimization_score"])

            row = {
                "method": "Bayesian Optimization",
                "trial": trial.number + 1,
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

            results.append(row)

            trial.set_user_attr("rule_count", len(rules))
            trial.set_user_attr("mining_time_seconds", mining_time)

            print(f"Rules generated: {len(rules)}")
            print(f"Optimization score: {score:.6f}")

            return score

        except Exception as error:

            print("Trial failed:", str(error))

            results.append({
                "method": "Bayesian Optimization",
                "trial": trial.number + 1,
                "min_support": min_support,
                "min_confidence": min_confidence,
                "max_len": max_len,
                "status": "Failed",
                "error": str(error),
                "test_set_used": False
            })

            raise

        finally:

            # Save progress after every trial.
            pd.DataFrame(results).to_csv(
                RESULT_FILE,
                index=False
            )

    # --------------------------------------------------------
    # 5. RUN BAYESIAN OPTIMIZATION
    # --------------------------------------------------------

    study.optimize(
        objective,
        n_trials=NUMBER_OF_TRIALS,
        catch=(Exception,)
    )

    # --------------------------------------------------------
    # 6. IDENTIFY BEST CONFIGURATION
    # --------------------------------------------------------

    completed_trials = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]

    failed_trials = NUMBER_OF_TRIALS - len(completed_trials)

    if not completed_trials:
        raise RuntimeError(
            "No Bayesian Optimization trial completed successfully."
        )

    best_trial = study.best_trial

    best_rows = [
        row for row in results
        if row["status"] == "Success"
        and row["trial"] == best_trial.number + 1
    ]

    best_metrics = best_rows[0]

    # --------------------------------------------------------
    # 7. SAVE BEST CONFIGURATION
    # --------------------------------------------------------

    best_output = {
        "method": "Bayesian Optimization",
        "optimization_algorithm": "Optuna TPE Sampler",
        "optuna_version": optuna.__version__,
        "random_state": RANDOM_STATE,
        "number_of_trials": NUMBER_OF_TRIALS,
        "successful_trials": len(completed_trials),
        "failed_trials": failed_trials,
        "best_configuration": best_metrics,
        "test_set_used": False
    }

    with open(BEST_FILE, "w", encoding="utf-8") as file:
        json.dump(best_output, file, indent=4)

    # --------------------------------------------------------
    # 8. DISPLAY RESULTS
    # --------------------------------------------------------

    total_time = time.time() - start_time

    results_df = pd.DataFrame(results)

    successful_results = results_df[
        results_df["status"] == "Success"
    ].copy()

    successful_results = successful_results.sort_values(
        by="optimization_score",
        ascending=False
    )

    print("\n" + "=" * 65)
    print("BAYESIAN OPTIMIZATION COMPLETED")
    print("=" * 65)

    print(f"\nSuccessful trials: {len(completed_trials)}")
    print(f"Failed trials: {failed_trials}")
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
