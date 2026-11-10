
import json
from pathlib import Path

import pandas as pd

from week5_utils import REPORT_DIR, create_output_directories


# ============================================================
# CARTWISE AI - WEEK 5 SUMMARY REPORT GENERATOR
# ============================================================

BASELINE_FILE = REPORT_DIR / "baseline_results.csv"
GRID_FILE = REPORT_DIR / "grid_search_results.csv"
RANDOM_FILE = REPORT_DIR / "random_search_results.csv"
BAYESIAN_FILE = REPORT_DIR / "bayesian_optimization_results.csv"

COMPARISON_FILE = REPORT_DIR / "optimization_comparison.csv"
FINAL_TEST_FILE = REPORT_DIR / "final_model_validation.json"

SUMMARY_CSV = REPORT_DIR / "week5_complete_summary.csv"
SUMMARY_JSON = REPORT_DIR / "week5_complete_summary.json"
SUMMARY_TXT = REPORT_DIR / "week5_complete_summary.txt"


def read_json(path):

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def count_trials(path):

    data = pd.read_csv(path)

    if "status" in data.columns:

        successful = (
            data["status"].astype(str).str.lower() == "success"
        ).sum()

        failed = len(data) - successful

    else:
        successful = len(data)
        failed = 0

    return {
        "total_trials": len(data),
        "successful_trials": int(successful),
        "failed_trials": int(failed)
    }


def main():

    create_output_directories()

    print("\n" + "=" * 65)
    print("CARTWISE AI - WEEK 5 COMPLETE SUMMARY")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. VERIFY REQUIRED FILES
    # --------------------------------------------------------

    required_files = [
        BASELINE_FILE,
        GRID_FILE,
        RANDOM_FILE,
        BAYESIAN_FILE,
        COMPARISON_FILE,
        FINAL_TEST_FILE
    ]

    for file_path in required_files:

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    # --------------------------------------------------------
    # 2. READ EXPERIMENT COUNTS
    # --------------------------------------------------------

    grid_counts = count_trials(GRID_FILE)
    random_counts = count_trials(RANDOM_FILE)
    bayesian_counts = count_trials(BAYESIAN_FILE)

    # --------------------------------------------------------
    # 3. READ COMPARISON RESULTS
    # --------------------------------------------------------

    comparison = pd.read_csv(COMPARISON_FILE)

    comparison = comparison.sort_values(
        by="optimization_score",
        ascending=False
    )

    baseline_score = float(
        comparison.loc[
            comparison["method"] == "Baseline",
            "optimization_score"
        ].iloc[0]
    )

    grid_best = comparison.loc[
        comparison["method"] == "Grid Search"
    ].iloc[0]

    # --------------------------------------------------------
    # 4. READ FINAL TEST RESULTS
    # --------------------------------------------------------

    final_test = read_json(FINAL_TEST_FILE)

    test_metrics = final_test["test_metrics"]

    # --------------------------------------------------------
    # 5. CALCULATE IMPROVEMENT
    # --------------------------------------------------------

    grid_score = float(grid_best["optimization_score"])

    improvement = (
        ((grid_score - baseline_score) / baseline_score) * 100
        if baseline_score != 0
        else 0
    )

    # --------------------------------------------------------
    # 6. BUILD SUMMARY
    # --------------------------------------------------------

    summary = {
        "project": "CartWise AI",
        "week": 5,
        "assignment": "Model Optimization and Experimentation",

        "dataset": {
            "total_transactions": 1910,
            "number_of_products": 40,
            "training_transactions": 1092,
            "validation_transactions": 466,
            "test_transactions": 352,
            "split_method": "Grouped split by identical basket patterns"
        },

        "experiments": {
            "baseline": {
                "method": "Baseline Evaluation",
                "configuration": {
                    "min_support": 0.01,
                    "min_confidence": 0.30,
                    "max_len": 3
                }
            },
            "grid_search": grid_counts,
            "random_search": random_counts,
            "bayesian_optimization": bayesian_counts
        },

        "selected_configuration": {
            "method": "Grid Search",
            "min_support": float(grid_best["min_support"]),
            "min_confidence": float(grid_best["min_confidence"]),
            "max_len": int(grid_best["max_len"]),
            "validation_score": grid_score,
            "baseline_score": baseline_score,
            "improvement_over_baseline_percent": improvement,
            "association_rules": int(grid_best["rule_count"])
        },

        "final_test_evaluation": {
            "precision_at_5": float(test_metrics["precision_at_5"]),
            "recall_at_5": float(test_metrics["recall_at_5"]),
            "hit_rate": float(test_metrics["hit_rate"]),
            "coverage": float(test_metrics["coverage"]),
            "budget_compliance": float(
                test_metrics["budget_compliance"]
            ),
            "average_spend": float(test_metrics["average_spend"]),
            "optimization_score": float(
                test_metrics["optimization_score"]
            ),
            "mean_lift": float(test_metrics["mean_lift"])
        },

        "test_set_used_during_tuning": False,

        "conclusion": (
            "Grid Search achieved the highest validation score "
            "jointly with Random Search and Bayesian Optimization. "
            "It was selected because it achieved the same score "
            "with fewer association rules. Final evaluation was "
            "then performed on the untouched test dataset."
        )
    }

    # --------------------------------------------------------
    # 7. SAVE CSV SUMMARY
    # --------------------------------------------------------

    summary_rows = [
        {
            "section": "Dataset",
            "metric": "Total transactions",
            "value": 1910
        },
        {
            "section": "Dataset",
            "metric": "Number of products",
            "value": 40
        },
        {
            "section": "Grid Search",
            "metric": "Successful configurations",
            "value": grid_counts["successful_trials"]
        },
        {
            "section": "Random Search",
            "metric": "Successful configurations",
            "value": random_counts["successful_trials"]
        },
        {
            "section": "Bayesian Optimization",
            "metric": "Successful configurations",
            "value": bayesian_counts["successful_trials"]
        },
        {
            "section": "Selected Model",
            "metric": "Minimum support",
            "value": grid_best["min_support"]
        },
        {
            "section": "Selected Model",
            "metric": "Minimum confidence",
            "value": grid_best["min_confidence"]
        },
        {
            "section": "Selected Model",
            "metric": "Maximum itemset length",
            "value": grid_best["max_len"]
        },
        {
            "section": "Final Test",
            "metric": "Precision@5",
            "value": test_metrics["precision_at_5"]
        },
        {
            "section": "Final Test",
            "metric": "Recall@5",
            "value": test_metrics["recall_at_5"]
        },
        {
            "section": "Final Test",
            "metric": "Hit Rate",
            "value": test_metrics["hit_rate"]
        },
        {
            "section": "Final Test",
            "metric": "Coverage",
            "value": test_metrics["coverage"]
        },
        {
            "section": "Final Test",
            "metric": "Budget Compliance",
            "value": test_metrics["budget_compliance"]
        },
        {
            "section": "Final Test",
            "metric": "Optimization Score",
            "value": test_metrics["optimization_score"]
        }
    ]

    pd.DataFrame(summary_rows).to_csv(
        SUMMARY_CSV,
        index=False
    )

    # --------------------------------------------------------
    # 8. SAVE JSON SUMMARY
    # --------------------------------------------------------

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(summary, file, indent=4)

    # --------------------------------------------------------
    # 9. CREATE HUMAN-READABLE TEXT REPORT
    # --------------------------------------------------------

    lines = [
        "CARTWISE AI - WEEK 5 SUMMARY",
        "=" * 60,
        "",
        "PROJECT: Model Optimization and Experimentation",
        "",
        "DATASET",
        "-" * 60,
        "Total transactions: 1910",
        "Products: 40",
        "Training transactions: 1092",
        "Validation transactions: 466",
        "Test transactions: 352",
        "",
        "OPTIMIZATION EXPERIMENTS",
        "-" * 60,
        f"Grid Search: {grid_counts['successful_trials']}/"
        f"{grid_counts['total_trials']} successful",
        f"Random Search: {random_counts['successful_trials']}/"
        f"{random_counts['total_trials']} successful",
        f"Bayesian Optimization: "
        f"{bayesian_counts['successful_trials']}/"
        f"{bayesian_counts['total_trials']} successful",
        "",
        "SELECTED CONFIGURATION",
        "-" * 60,
        "Method: Grid Search",
        f"Minimum support: {grid_best['min_support']}",
        f"Minimum confidence: {grid_best['min_confidence']}",
        f"Maximum itemset length: {grid_best['max_len']}",
        f"Association rules: {int(grid_best['rule_count'])}",
        f"Validation score: {grid_score:.6f}",
        f"Improvement over baseline: {improvement:.2f}%",
        "",
        "FINAL TEST RESULTS",
        "-" * 60,
        f"Precision@5: {test_metrics['precision_at_5']:.6f}",
        f"Recall@5: {test_metrics['recall_at_5']:.6f}",
        f"Hit Rate: {test_metrics['hit_rate']:.6f}",
        f"Coverage: {test_metrics['coverage']:.6f}",
        f"Budget compliance: "
        f"{test_metrics['budget_compliance']:.6f}",
        f"Optimization score: "
        f"{test_metrics['optimization_score']:.6f}",
        "",
        "TEST SET WAS NOT USED DURING HYPERPARAMETER TUNING.",
        "",
        "CONCLUSION",
        "-" * 60,
        summary["conclusion"],
        ""
    ]

    SUMMARY_TXT.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # 10. DISPLAY RESULTS
    # --------------------------------------------------------

    print("\nEXPERIMENT COUNTS")

    print("Grid Search:", grid_counts)
    print("Random Search:", random_counts)
    print("Bayesian Optimization:", bayesian_counts)

    print("\nSELECTED CONFIGURATION")
    print("Method: Grid Search")
    print(f"Minimum support: {grid_best['min_support']}")
    print(f"Minimum confidence: {grid_best['min_confidence']}")
    print(f"Maximum length: {grid_best['max_len']}")
    print(f"Validation score: {grid_score:.6f}")

    print("\nFINAL TEST RESULTS")

    for metric, value in summary["final_test_evaluation"].items():
        print(f"{metric}: {value:.6f}")

    print("\nSaved files:")
    print(SUMMARY_CSV)
    print(SUMMARY_JSON)
    print(SUMMARY_TXT)

    print("\nWEEK 5 SUMMARY GENERATED SUCCESSFULLY")


if __name__ == "__main__":
    main()
