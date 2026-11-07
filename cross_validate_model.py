
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from mlxtend.frequent_patterns import apriori, association_rules


# ============================================================
# CARTWISE AI - WEEK 4
# FIVE-FOLD CROSS-VALIDATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

BASKET_FILE = BASE_DIR / "data/processed/basket_matrix.csv"
REPORT_DIR = BASE_DIR / "data/reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42

MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.30
MAX_ITEMSET_LENGTH = 3


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 65)
print("CARTWISE AI - FIVE-FOLD CROSS-VALIDATION")
print("=" * 65)

basket = pd.read_csv(BASKET_FILE, index_col=0)
basket = basket.fillna(False).astype(bool)

print("\nDATASET INFORMATION")
print("-" * 40)
print("Transactions:", len(basket))
print("Products:", basket.shape[1])


# ============================================================
# 2. INITIALIZE CROSS-VALIDATION
# ============================================================

kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

fold_results = []
fold_rule_results = []

# Track how often each association rule is discovered.
rule_occurrences = {}


# ============================================================
# 3. TRAIN AND VALIDATE EACH FOLD
# ============================================================

for fold_number, (train_indices, test_indices) in enumerate(
    kf.split(basket), start=1
):

    print("\n" + "=" * 65)
    print(f"FOLD {fold_number}")
    print("=" * 65)

    train_data = basket.iloc[train_indices].reset_index(drop=True)
    test_data = basket.iloc[test_indices].reset_index(drop=True)

    print("Training transactions:", len(train_data))
    print("Validation transactions:", len(test_data))

    # --------------------------------------------------------
    # Mine frequent itemsets using training data only
    # --------------------------------------------------------

    frequent_itemsets = apriori(
        train_data,
        min_support=MIN_SUPPORT,
        use_colnames=True,
        max_len=MAX_ITEMSET_LENGTH,
        low_memory=True
    )

    if frequent_itemsets.empty:
        print("No frequent itemsets found in this fold.")
        continue

    # --------------------------------------------------------
    # Generate association rules
    # --------------------------------------------------------

    rule_args = {
        "metric": "confidence",
        "min_threshold": MIN_CONFIDENCE
    }

    if "num_itemsets" in inspect.signature(association_rules).parameters:
        rule_args["num_itemsets"] = len(train_data)

    rules = association_rules(
        frequent_itemsets,
        **rule_args
    )

    if rules.empty:
        print("No association rules found in this fold.")
        continue

    # --------------------------------------------------------
    # Evaluate rules on validation data
    # --------------------------------------------------------

    validation_confidences = []
    validation_lifts = []
    validation_supports = []

    for _, rule in rules.iterrows():

        antecedents = set(rule["antecedents"])
        consequents = set(rule["consequents"])
        combined = antecedents | consequents

        antecedent_count = int(
            test_data[list(antecedents)].all(axis=1).sum()
        )

        consequent_count = int(
            test_data[list(consequents)].all(axis=1).sum()
        )

        combined_count = int(
            test_data[list(combined)].all(axis=1).sum()
        )

        test_support = combined_count / len(test_data)

        test_confidence = (
            combined_count / antecedent_count
            if antecedent_count > 0 else np.nan
        )

        consequent_support = (
            consequent_count / len(test_data)
        )

        test_lift = (
            test_confidence / consequent_support
            if consequent_support > 0
            and not np.isnan(test_confidence)
            else np.nan
        )

        validation_supports.append(test_support)

        if not np.isnan(test_confidence):
            validation_confidences.append(test_confidence)

        if not np.isnan(test_lift):
            validation_lifts.append(test_lift)

        rule_key = (
            tuple(sorted(antecedents)),
            tuple(sorted(consequents))
        )

        rule_occurrences[rule_key] = (
            rule_occurrences.get(rule_key, 0) + 1
        )

        fold_rule_results.append({
            "fold": fold_number,
            "antecedents": ", ".join(sorted(antecedents)),
            "consequents": ", ".join(sorted(consequents)),
            "train_support": rule["support"],
            "train_confidence": rule["confidence"],
            "train_lift": rule["lift"],
            "validation_support": test_support,
            "validation_confidence": test_confidence,
            "validation_lift": test_lift
        })

    # --------------------------------------------------------
    # Store fold-level results
    # --------------------------------------------------------

    fold_result = {
        "fold": fold_number,
        "training_transactions": len(train_data),
        "validation_transactions": len(test_data),
        "frequent_itemsets": len(frequent_itemsets),
        "association_rules": len(rules),
        "mean_validation_support": (
            np.mean(validation_supports)
            if validation_supports else np.nan
        ),
        "mean_validation_confidence": (
            np.mean(validation_confidences)
            if validation_confidences else np.nan
        ),
        "mean_validation_lift": (
            np.mean(validation_lifts)
            if validation_lifts else np.nan
        )
    }

    fold_results.append(fold_result)

    print("Frequent itemsets:", len(frequent_itemsets))
    print("Association rules:", len(rules))
    print(
        "Mean validation confidence:",
        round(fold_result["mean_validation_confidence"], 4)
    )
    print(
        "Mean validation lift:",
        round(fold_result["mean_validation_lift"], 4)
    )


# ============================================================
# 4. SAVE CROSS-VALIDATION RESULTS
# ============================================================

if len(fold_results) != N_SPLITS:
    raise RuntimeError(
        "Not all five folds produced association rules. "
        "Inspect the fold outputs before interpreting results."
    )

fold_report = pd.DataFrame(fold_results)

fold_report.to_csv(
    REPORT_DIR / "cross_validation_results.csv",
    index=False
)

rule_report = pd.DataFrame(fold_rule_results)

rule_report.to_csv(
    REPORT_DIR / "cross_validation_rule_metrics.csv",
    index=False
)


# ============================================================
# 5. CALCULATE MEAN AND STANDARD DEVIATION
# ============================================================

metric_columns = [
    "frequent_itemsets",
    "association_rules",
    "mean_validation_support",
    "mean_validation_confidence",
    "mean_validation_lift"
]

summary_rows = []

for metric in metric_columns:

    values = fold_report[metric].dropna()

    summary_rows.append({
        "metric": metric,
        "mean": values.mean(),
        "standard_deviation": values.std(ddof=1)
    })

summary_report = pd.DataFrame(summary_rows)

summary_report.to_csv(
    REPORT_DIR / "cross_validation_summary.csv",
    index=False
)


# ============================================================
# 6. CALCULATE RULE STABILITY
# ============================================================

stability_rows = []

for rule_key, count in rule_occurrences.items():

    antecedents, consequents = rule_key

    stability_rows.append({
        "antecedents": ", ".join(antecedents),
        "consequents": ", ".join(consequents),
        "folds_found": count,
        "stability_percentage": (
            count / N_SPLITS
        ) * 100
    })

stability_report = pd.DataFrame(stability_rows)

stability_report = stability_report.sort_values(
    ["stability_percentage", "folds_found"],
    ascending=False
)

stability_report.to_csv(
    REPORT_DIR / "rule_stability.csv",
    index=False
)


# ============================================================
# 7. DISPLAY FINAL RESULTS
# ============================================================

print("\n" + "=" * 65)
print("CROSS-VALIDATION SUMMARY")
print("=" * 65)

print(fold_report.round(4).to_string(index=False))

print("\nMEAN AND STANDARD DEVIATION")
print("-" * 65)
print(summary_report.round(4).to_string(index=False))

print("\nRULE STABILITY")
print("-" * 65)
print("Unique rules discovered:", len(stability_report))
print(
    "Rules appearing in all five folds:",
    int((stability_report["folds_found"] == N_SPLITS).sum())
)

print("\nOUTPUT FILES")
print("-" * 40)
print(REPORT_DIR / "cross_validation_results.csv")
print(REPORT_DIR / "cross_validation_rule_metrics.csv")
print(REPORT_DIR / "cross_validation_summary.csv")
print(REPORT_DIR / "rule_stability.csv")

print("\nFIVE-FOLD CROSS-VALIDATION COMPLETED")
print("=" * 65)
