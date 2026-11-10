
from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt

from week5_utils import REPORT_DIR, VISUALIZATION_DIR, create_output_directories


# ============================================================
# CARTWISE AI - WEEK 5 OPTIMIZATION COMPARISON
# ============================================================

BASELINE_FILE = REPORT_DIR / "baseline_results.csv"
GRID_FILE = REPORT_DIR / "grid_search_results.csv"
RANDOM_FILE = REPORT_DIR / "random_search_results.csv"
BAYESIAN_FILE = REPORT_DIR / "bayesian_optimization_results.csv"

OUTPUT_FILE = REPORT_DIR / "optimization_comparison.csv"
SUMMARY_FILE = REPORT_DIR / "optimization_comparison_summary.json"

CHART_FILE = VISUALIZATION_DIR / "optimization_method_comparison.png"


def load_best_result(file_path, method, baseline=False):

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required result file not found: {file_path}"
        )

    data = pd.read_csv(file_path)

    if data.empty:
        raise ValueError(f"No results found in {file_path}")

    if "optimization_score" not in data.columns:
        raise ValueError(
            f"optimization_score column missing in {file_path}"
        )

    if "status" in data.columns:
        data = data[
            data["status"].astype(str).str.lower() == "success"
        ].copy()

    data["optimization_score"] = pd.to_numeric(
        data["optimization_score"],
        errors="coerce"
    )

    data = data.dropna(subset=["optimization_score"])

    if data.empty:
        raise ValueError(
            f"No successful configurations found in {file_path}"
        )

    if baseline:
        best = data.iloc[0]
    else:
        best = data.loc[data["optimization_score"].idxmax()]

    result = best.to_dict()
    result["method"] = method

    return result


def main():

    create_output_directories()

    print("\n" + "=" * 65)
    print("CARTWISE AI - OPTIMIZATION METHOD COMPARISON")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. LOAD BEST RESULTS FROM EACH METHOD
    # --------------------------------------------------------

    methods = [
        (
            "Baseline",
            BASELINE_FILE,
            True
        ),
        (
            "Grid Search",
            GRID_FILE,
            False
        ),
        (
            "Random Search",
            RANDOM_FILE,
            False
        ),
        (
            "Bayesian Optimization",
            BAYESIAN_FILE,
            False
        )
    ]

    comparison_results = []

    for method, file_path, is_baseline in methods:

        print(f"\nLoading {method} results...")

        result = load_best_result(
            file_path,
            method,
            baseline=is_baseline
        )

        comparison_results.append(result)

        print(
            f"Best optimization score: "
            f"{result['optimization_score']:.6f}"
        )

    # --------------------------------------------------------
    # 2. CREATE COMPARISON TABLE
    # --------------------------------------------------------

    comparison_df = pd.DataFrame(comparison_results)

    important_columns = [
        "method",
        "min_support",
        "min_confidence",
        "max_len",
        "optimization_score",
        "precision_at_5",
        "recall_at_5",
        "hit_rate",
        "coverage",
        "budget_compliance",
        "rule_count",
        "mining_time_seconds",
        "evaluation_time_seconds"
    ]

    available_columns = [
        column
        for column in important_columns
        if column in comparison_df.columns
    ]

    comparison_df = comparison_df[available_columns]

    comparison_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # 3. CALCULATE IMPROVEMENT OVER BASELINE
    # --------------------------------------------------------

    baseline_score = float(
        comparison_df.loc[
            comparison_df["method"] == "Baseline",
            "optimization_score"
        ].iloc[0]
    )

    summary = {
        "project": "CartWise AI",
        "week": 5,
        "comparison_methods": [],
        "baseline_score": baseline_score,
        "test_set_used": False
    }

    for _, row in comparison_df.iterrows():

        score = float(row["optimization_score"])

        improvement = (
            ((score - baseline_score) / baseline_score) * 100
            if baseline_score != 0
            else 0
        )

        summary["comparison_methods"].append({
            "method": row["method"],
            "optimization_score": score,
            "improvement_over_baseline_percent": improvement,
            "precision_at_5": float(row["precision_at_5"]),
            "recall_at_5": float(row["recall_at_5"]),
            "hit_rate": float(row["hit_rate"]),
            "coverage": float(row["coverage"]),
            "rule_count": (
                int(row["rule_count"])
                if pd.notna(row.get("rule_count"))
                else None
            )
        })

    # Select the most efficient method among those
    # tied for the highest optimization score.
    best_score = comparison_df["optimization_score"].max()

    tied_results = comparison_df[
        comparison_df["optimization_score"] == best_score
    ].copy()

    if "rule_count" in tied_results.columns:
        tied_results["rule_count"] = pd.to_numeric(
            tied_results["rule_count"],
            errors="coerce"
        )
        tied_results = tied_results.sort_values(
            by="rule_count",
            ascending=True
        )

    best_method = tied_results.iloc[0]

    summary["highest_scoring_method"] = comparison_df.loc[
        comparison_df["optimization_score"].idxmax(),
        "method"
    ]

    summary["most_efficient_tied_method"] = best_method["method"]

    with open(SUMMARY_FILE, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    # --------------------------------------------------------
    # 4. GENERATE OPTIMIZATION SCORE CHART
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.bar(
        comparison_df["method"],
        comparison_df["optimization_score"]
    )

    plt.title("CartWise AI - Optimization Score Comparison")
    plt.xlabel("Optimization Method")
    plt.ylabel("Optimization Score")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        CHART_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # 5. DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("OPTIMIZATION COMPARISON RESULTS")
    print("=" * 65)

    print(comparison_df.to_string(index=False))

    print("\nBEST SCORING METHOD:")
    print(summary["highest_scoring_method"])

    print("\nMOST EFFICIENT METHOD AMONG TOP-SCORING TIES:")
    print(summary["most_efficient_tied_method"])

    print("\nIMPROVEMENT OVER BASELINE:")

    for item in summary["comparison_methods"]:
        print(
            f"{item['method']}: "
            f"{item['improvement_over_baseline_percent']:.2f}%"
        )

    print("\nSaved files:")
    print(OUTPUT_FILE)
    print(SUMMARY_FILE)
    print(CHART_FILE)

    print(
        "\nNote: These are validation results. "
        "The untouched test set has not been evaluated."
    )


if __name__ == "__main__":
    main()
