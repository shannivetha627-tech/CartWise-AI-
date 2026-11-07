
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# CARTWISE AI - WEEK 4 CONSOLIDATED EVALUATION SUMMARY
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

REPORT_DIR = BASE_DIR / "data" / "reports"
VISUAL_DIR = BASE_DIR / "visualizations"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
VISUAL_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------
# 1. LOAD REPORTS
# ----------------------------------------------------------

def load_reports():

    print("\nLoading Week 4 reports...")

    filenames = {
        "train_test_rules": "train_test_rule_evaluation.csv",
        "train_test_cases": "train_test_case_results.csv",
        "recommendations": "train_test_recommendation_metrics.csv",
        "cv_results": "cross_validation_results.csv",
        "cv_summary": "cross_validation_summary.csv",
        "rule_stability": "rule_stability.csv",
        "error_summary": "error_analysis_summary.csv",
        "product_errors": "product_error_analysis.csv"
    }

    reports = {}

    for key, filename in filenames.items():

        path = REPORT_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing report: {path}")

        reports[key] = pd.read_csv(path)

        print("Loaded:", filename)

    return reports


# ----------------------------------------------------------
# 2. TRAIN-TEST RULE EVALUATION
# ----------------------------------------------------------

def extract_train_test_metrics(reports):

    df = reports["train_test_rules"]

    metrics = []

    metric_columns = [
        "train_support",
        "train_confidence",
        "train_lift",
        "test_support",
        "test_confidence",
        "test_lift",
        "test_leverage",
        "test_conviction"
    ]

    for column in metric_columns:

        if column in df.columns:

            values = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            metrics.append({
                "evaluation_method": "Train-Test Rule Evaluation",
                "metric": f"mean_{column}",
                "value": values.mean()
            })

    if "antecedents" in df.columns:

        metrics.append({
            "evaluation_method": "Train-Test Rule Evaluation",
            "metric": "total_evaluated_rules",
            "value": len(df)
        })

    return metrics


# ----------------------------------------------------------
# 3. RECOMMENDATION EVALUATION
# ----------------------------------------------------------

def extract_recommendation_metrics(reports):

    df = reports["recommendations"]

    if df.empty:
        return []

    row = df.iloc[0]

    metrics = []

    for column in df.columns:

        if pd.api.types.is_numeric_dtype(df[column]):

            metrics.append({
                "evaluation_method": "Recommendation Evaluation",
                "metric": column,
                "value": row[column]
            })

    return metrics


# ----------------------------------------------------------
# 4. CROSS-VALIDATION SUMMARY
# ----------------------------------------------------------

def extract_cv_metrics(reports):

    df = reports["cv_summary"]

    metrics = []

    for _, row in df.iterrows():

        metric_name = row["metric"]

        metrics.append({
            "evaluation_method": "Five-Fold Cross-Validation",
            "metric": f"mean_{metric_name}",
            "value": row["mean"]
        })

        metrics.append({
            "evaluation_method": "Five-Fold Cross-Validation",
            "metric": f"std_{metric_name}",
            "value": row["standard_deviation"]
        })

    return metrics


# ----------------------------------------------------------
# 5. RULE STABILITY
# ----------------------------------------------------------

def extract_stability_metrics(reports):

    df = reports["rule_stability"]

    metrics = []

    if df.empty:
        return metrics

    if "frequency" in df.columns:

        metrics.append({
            "evaluation_method": "Rule Stability",
            "metric": "mean_rule_fold_frequency",
            "value": df["frequency"].mean()
        })

    if "frequency" in df.columns:

        metrics.append({
            "evaluation_method": "Rule Stability",
            "metric": "rules_in_all_five_folds",
            "value": (df["frequency"] == 5).sum()
        })

    return metrics


# ----------------------------------------------------------
# 6. ERROR ANALYSIS
# ----------------------------------------------------------

def extract_error_metrics(reports):

    df = reports["error_summary"]

    if df.empty:
        return []

    row = df.iloc[0]

    metrics = []

    for column in df.columns:

        if pd.api.types.is_numeric_dtype(df[column]):

            metrics.append({
                "evaluation_method": "Recommendation Error Analysis",
                "metric": column,
                "value": row[column]
            })

    return metrics


# ----------------------------------------------------------
# 7. CONSOLIDATE ALL METRICS
# ----------------------------------------------------------

def create_summary(reports):

    all_metrics = []

    all_metrics.extend(
        extract_train_test_metrics(reports)
    )

    all_metrics.extend(
        extract_recommendation_metrics(reports)
    )

    all_metrics.extend(
        extract_cv_metrics(reports)
    )

    all_metrics.extend(
        extract_stability_metrics(reports)
    )

    all_metrics.extend(
        extract_error_metrics(reports)
    )

    summary = pd.DataFrame(all_metrics)

    output_file = (
        REPORT_DIR / "week4_consolidated_summary.csv"
    )

    summary.to_csv(
        output_file,
        index=False
    )

    print("\nCONSOLIDATED WEEK 4 METRICS")

    print(summary.to_string(index=False))

    print("\nSaved:", output_file)

    return summary


# ----------------------------------------------------------
# 8. CROSS-VALIDATION VISUALIZATION
# ----------------------------------------------------------

def create_cv_chart(reports):

    df = reports["cv_results"]

    required = [
        "fold",
        "mean_validation_confidence",
        "mean_validation_lift"
    ]

    if not all(column in df.columns for column in required):

        print("\nCross-validation chart skipped: columns unavailable.")

        return

    plt.figure(figsize=(9, 5))

    plt.plot(
        df["fold"],
        df["mean_validation_confidence"],
        marker="o",
        label="Validation Confidence"
    )

    plt.plot(
        df["fold"],
        df["mean_validation_lift"],
        marker="s",
        label="Validation Lift"
    )

    plt.xlabel("Fold")
    plt.ylabel("Metric Value")

    plt.title(
        "CartWise AI - Five-Fold Cross-Validation"
    )

    plt.xticks(df["fold"])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output = VISUAL_DIR / "week4_cross_validation.png"

    plt.savefig(output, dpi=300)
    plt.close()

    print("\nSaved:", output)


# ----------------------------------------------------------
# 9. ERROR COMPARISON VISUALIZATION
# ----------------------------------------------------------

def create_error_chart(reports):

    df = reports["error_summary"]

    row = df.iloc[0]

    labels = [
        "True Positives",
        "False Positives",
        "False Negatives"
    ]

    values = [
        row["true_positives"],
        row["false_positives"],
        row["false_negatives"]
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(labels, values)

    plt.title(
        "CartWise AI - Recommendation Error Analysis"
    )

    plt.ylabel("Number of Products")

    plt.tight_layout()

    output = VISUAL_DIR / "week4_error_comparison.png"

    plt.savefig(output, dpi=300)
    plt.close()

    print("\nSaved:", output)


# ----------------------------------------------------------
# 10. MAIN
# ----------------------------------------------------------

def main():

    print("=" * 60)

    print(
        "CARTWISE AI - WEEK 4 CONSOLIDATED SUMMARY"
    )

    print("=" * 60)

    reports = load_reports()

    create_summary(reports)

    create_cv_chart(reports)

    create_error_chart(reports)

    print("\nWEEK 4 CONSOLIDATED SUMMARY COMPLETED")


if __name__ == "__main__":

    main()
