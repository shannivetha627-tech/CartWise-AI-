
import pandas as pd
import inspect
from pathlib import Path
from mlxtend.frequent_patterns import apriori, association_rules

# Load the Boolean transaction matrix
input_file = "data/processed/basket_matrix.csv"

basket = pd.read_csv(input_file, index_col=0)
basket = basket.fillna(False).astype(bool)

print("=" * 55)
print("CARTWISE AI - APRIORI ASSOCIATION RULE MINING")
print("=" * 55)

print("Number of transactions:", basket.shape[0])
print("Number of products:", basket.shape[1])

# Generate frequent itemsets
frequent_itemsets = apriori(
    basket,
    min_support=0.01,
    use_colnames=True,
    max_len=3,
    low_memory=True
)

# Generate association rules
if not frequent_itemsets.empty:
    parameters = inspect.signature(association_rules).parameters

    rule_args = {
        "metric": "confidence",
        "min_threshold": 0.30
    }

    # Compatibility with mlxtend versions requiring num_itemsets
    if "num_itemsets" in parameters:
        rule_args["num_itemsets"] = len(basket)

    rules = association_rules(
        frequent_itemsets,
        **rule_args
    )

    if not rules.empty:
        rules = rules.sort_values(
            by="lift",
            ascending=False
        )
else:
    rules = pd.DataFrame()

# Create output directory
output_dir = Path("data/reports")
output_dir.mkdir(parents=True, exist_ok=True)

# Save frequent itemsets
if not frequent_itemsets.empty:
    itemsets_to_save = frequent_itemsets.copy()
    itemsets_to_save["itemsets"] = itemsets_to_save["itemsets"].apply(
        lambda items: ", ".join(sorted(map(str, items)))
    )
else:
    itemsets_to_save = frequent_itemsets

itemsets_to_save.to_csv(
    output_dir / "apriori_frequent_itemsets.csv",
    index=False
)

# Save association rules
if not rules.empty:
    rules_to_save = rules.copy()

    for column in ["antecedents", "consequents"]:
        rules_to_save[column] = rules_to_save[column].apply(
            lambda items: ", ".join(sorted(map(str, items)))
        )
else:
    rules_to_save = rules

rules_to_save.to_csv(
    output_dir / "apriori_association_rules.csv",
    index=False
)

# Display results
print("\nFrequent itemsets found:", len(frequent_itemsets))
print("Association rules found:", len(rules))

if not rules.empty:
    print("\nTOP 10 RULES BY LIFT")
    print(
        rules[
            ["antecedents", "consequents",
             "support", "confidence", "lift"]
        ].head(10).to_string(index=False)
    )
else:
    print("\nNo rules found with the current thresholds.")

print("\nOutput files:")
print(output_dir / "apriori_frequent_itemsets.csv")
print(output_dir / "apriori_association_rules.csv")

print("\nAPRIORI MINING COMPLETED")
