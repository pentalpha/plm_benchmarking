import os
import sys
import json

import ast
from sklearn.tree import DecisionTreeClassifier, export_text
from mlxtend.frequent_patterns import fpgrowth, association_rules
import pandas as pd


if __name__ == "__main__":
    benchmarking_dir = sys.argv[1]
    results_df = pd.read_csv(f"{benchmarking_dir}/results_all.tsv", sep="\t")

    model_sizes = json.load(open("input_data/model_sizes.json"))

    #Create categorical performance indicator by ontology
    ont_min = {}
    ont_max = {}

    for ont, ont_df in results_df.groupby("ontology"):
        ont_min[ont] = ont_df[ont_df["Sort Score"]> 0]["Sort Score"].min()
        ont_max[ont] = ont_df["Sort Score"].max()
    
    perf_cats = ["Mediocre", "Medium High", "High"]
    bins = [0.8, 0.9]

    def assign_cat(val, ont):
        if val < 0:
            return "N/A"
        cat_span = ont_max[ont] - ont_min[ont]
        norm_val = (val - ont_min[ont]) / cat_span
        for i, b in enumerate(bins):
            if norm_val < b:
                return perf_cats[i]
        return perf_cats[-1]
        
    results_df["Category"] = results_df.apply(lambda row: assign_cat(row["Sort Score"], row["ontology"]), axis=1)
    results_df["is_nan"] = results_df["Sort Score"] < 0

    for ont in results_df["ontology"].unique():
        example = results_df[["ontology", "Sort Score", "Category"]][results_df["ontology"] == ont]
        example = example[example["Sort Score"] > 0]
        print(example)
    
    print(results_df)
    print("\nColumn\tN_unique\tis_numeric")
    for col in results_df:
        print(col, results_df[col].nunique(), pd.api.types.is_numeric_dtype(results_df[col]))

    print("\nStarting Pattern Mining Analysis...")

    # Identify Py-Boost parameter columns + embedding size flag
    param_cols = [col for col in results_df.columns if col.startswith("param_")] + ["large_emb"]
    param_cols.remove("param_comb_id")
    
    binned_df = results_df.copy()
    for col in param_cols:
        # Group continuous metrics (like lr or lambda_l2) into buckets
        if pd.api.types.is_numeric_dtype(binned_df[col]) and binned_df[col].nunique() > 5:
            # THE FIX: Remove explicit labels. Pandas will generate string intervals like "(0.01, 0.05]"
            binned_df[f"{col}_binned"] = pd.qcut(
                binned_df[col], q=3, duplicates='drop'
            ).astype(str)
        else:
            binned_df[f"{col}_binned"] = binned_df[col].astype(str)

    binned_param_cols = [col for col in binned_df.columns if col.endswith("_binned")]
    
    # --- 2. OOM (NaN) ERROR MINING ---
    '''print("\n--- ANALYZING OOM (NaN) ERRORS ---")
    
    df_nans = results_df[param_cols].fillna(0)
    X_nan = pd.get_dummies(df_nans, drop_first=True)
    y_nan = results_df["is_nan"]
    
    if y_nan.sum() > 0:
        dtree_nan = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
        dtree_nan.fit(X_nan, y_nan)
        print("Decision Tree Rules for NaNs (1 = OOM Error, 0 = Valid Execution):")
        print(export_text(dtree_nan, feature_names=list(X_nan.columns)))

    apriori_nan_df = pd.get_dummies(binned_df[binned_param_cols + ["is_nan"]]).astype(bool)
    
    freq_items_nan = fpgrowth(apriori_nan_df, min_support=0.015, use_colnames=True)
    
    if not freq_items_nan.empty:
        # LOWERED LIFT TO 1.05
        rules_nan = association_rules(freq_items_nan, metric="lift", min_threshold=1.05)
        oom_rules = rules_nan[rules_nan['consequents'] == frozenset({'is_nan'})] 
        oom_rules = oom_rules.sort_values(by=['confidence', 'lift'], ascending=[False, False])
        print("\nTop Parameter Tuples Associated with NaN (OOM):")
        print(oom_rules[['antecedents', 'confidence', 'lift']].head(10))'''

    # --- 3. HIGH & LOW PERFORMANCE MINING ---
    print("\n--- ANALYZING PERFORMANCE TUPLES ---")

    all_rules = []

    for is_large, group_df in binned_df.groupby("large_emb"):
        print(f"\nAnalyzing parameters for large_emb = {is_large}")
        valid_df = group_df[group_df["is_nan"] == False]
    
        cols = binned_param_cols + ["Category"] 
        apriori_perf_df = pd.get_dummies(valid_df[cols]).astype(bool)
        
        freq_items_perf = fpgrowth(apriori_perf_df, min_support=0.015, use_colnames=True, max_len=4)
        
        if not freq_items_perf.empty:
            rules_perf = association_rules(freq_items_perf, metric="lift", min_threshold=1.05)
            rules_perf["large_emb"] = is_large

            # FIX: Convert frozensets to clean comma-separated strings
            rules_perf['antecedents'] = rules_perf['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules_perf['consequents'] = rules_perf['consequents'].apply(lambda x: ', '.join(list(x)))

            # FIX: Search for strings, and copy to prevent Pandas warnings
            high_rules = rules_perf[rules_perf['consequents'] == 'Category_High'].copy()
            high_rules = high_rules.sort_values(by=['confidence', 'lift'], ascending=[False, False])
            print("\nTop Parameter Tuples Associated with HIGH Performance:")
            print(high_rules[['antecedents', 'confidence', 'lift']].head(10))

            # FIX: Search for strings, and copy to prevent Pandas warnings
            low_rules = rules_perf[rules_perf['consequents'] == 'Category_Mediocre'].copy()
            low_rules = low_rules.sort_values(by=['confidence', 'lift'], ascending=[False, False])
            print("\nTop Parameter Tuples Associated with LOW Performance:")
            print(low_rules[['antecedents', 'confidence', 'lift']].head(10))
            
            # FIX: Append the ENTIRE unfettered ruleset to the file buffer, not just the High/Low head
            valid_input = rules_perf[~rules_perf["antecedents"].str.contains("Category_")]
            valid_output = valid_input[valid_input["consequents"].isin(["Category_High", "Category_Mediocre"])]
            all_rules.append(valid_output)
    
    rules_df = pd.concat(all_rules)
    
    # Reorder columns to put antecedents and consequents at the front of the TSV
    cols = ['antecedents', 'consequents'] + [c for c in rules_df.columns if c not in ['antecedents', 'consequents']]
    rules_df = rules_df[cols]
    rules_df.sort_values(by=["confidence", "lift"], ascending=[False, False], inplace=True)

    metric_cols = ["antecedent support", "consequent support", "support", "confidence", "lift", "representativity", "leverage", "conviction", "zhangs_metric", "jaccard", "certainty", "kulczynski"]
    for col in metric_cols:
        print(col)
        print(rules_df[col].describe())
    
    rules_df.to_csv(f"{benchmarking_dir}/rules.tsv", sep="\t", index=False)

    # Calculate how many parameters are in each rule
    rules_df['rule_length'] = rules_df['antecedents'].astype(str).apply(lambda x: len(x.split(',')))
    
    def get_top_rules(df, target, emb_size, max_params=2, top_n=5):
        # Filter by outcome, embedding size, and rule complexity
        subset = df[
            (df['consequents'] == target) & 
            (df['antecedents'].str.contains(f"large_emb_binned_{emb_size}")) &
            (df['rule_length'] <= (max_params + 1)) # +1 because large_emb is counted as a parameter
        ].copy()
        
        # Sort by Lift (correlation strength) and Confidence (reliability)
        subset = subset.sort_values(by=['lift', 'confidence'], ascending=[False, False])
        
        # Clean up the output for readability
        subset['antecedents'] = subset['antecedents'].str.replace(f"large_emb_binned_{emb_size}, ", "")
        subset['antecedents'] = subset['antecedents'].str.replace(f", large_emb_binned_{emb_size}", "")
        
        return subset[['antecedents', 'confidence', 'lift']].head(top_n)

    all_tops = []

    print("\n--- SINGLE MOST IMPORTANT PARAMETERS ---")
    print("\nLARGE Models -> HIGH Performance:")
    top_1 = get_top_rules(rules_df, 'Category_High', 'True', max_params=1, top_n=6)
    top_1["Ruleset"] = "Top 6 Important Parameters - Large Models High"
    all_tops.append(top_1)
    print(top_1)
    
    print("\nSMALL Models -> HIGH Performance:")
    top_2 = get_top_rules(rules_df, 'Category_High', 'False', max_params=1, top_n=6)
    top_2["Ruleset"] = "Top 6 Important Parameters - Small Models High"
    all_tops.append(top_2)
    print(top_2)

    print("\n--- DEADLY SINS (Single Parameters causing Mediocre outcomes) ---")
    print("\nLARGE Models -> MEDIOCRE Performance:")
    top_3 = get_top_rules(rules_df, 'Category_Mediocre', 'True', max_params=1, top_n=6)
    top_3["Ruleset"] = "Top 6 Deadly Sins - Large Models Mediocre"
    all_tops.append(top_3)
    print(top_3)
    
    print("\n--- BEST COMBINATIONS (Pairs of Parameters) ---")
    print("\nLARGE Models -> HIGH Performance:")
    top_4 = get_top_rules(rules_df, 'Category_High', 'True', max_params=2, top_n=12)
    top_4["Ruleset"] = "Top 12 Best Combinations - Large Models High"
    all_tops.append(top_4)
    print(top_4)
    
    all_tops_df = pd.concat(all_tops)
    all_tops_df.to_csv(f"{benchmarking_dir}/top_rules.tsv", sep="\t", index=False)