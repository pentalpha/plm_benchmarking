import sys
import glob
import os
import json
from cupy import full
import polars as pl

output_path = sys.argv[1]
result_dirs = sys.argv[2:]

eval_prefixes = [
    ("CAFA Fmax Macro", "fmax_col_mean_cafa"),
    ("NAN Fmax Macro", "fmax_col_mean_nan"),
]
training_types = ["u900", "composite", "more_negatives", ""]
model_names_base = [
    "NaN Masking",
    "Classic+NaN Masking+Neg",
    "NaN Masking+Neg",
    "Binary Targets",
]
model_results = {}

for dir_path in result_dirs:
    print(dir_path)
    if "fuzzy-" in dir_path:
        target_type = "Fuzzy Values"
    elif "u900-" in dir_path:
        target_type = "u900"
    elif "classic-" in dir_path:
        target_type = "Classic (Full Binary)"
    else:
        target_type = "unknown"
    stats_path = os.path.join(dir_path, "statistics.json")
    if not os.path.exists(stats_path):
        print(f"Stats path not found: {stats_path}")
        continue
    stats = json.load(open(stats_path))
    print(stats.keys())

    param_comb_path = os.path.join(dir_path, "param_comb.json")
    param_comb = json.load(open(param_comb_path))

    for model_name, model_type in zip(model_names_base, training_types):
        fmax_values = {}
        for eval_name, eval_type in eval_prefixes:
            key_name = (eval_type + "_" + model_type).strip("_")
            if key_name in stats:
                fmax_value = stats[key_name]
                fmax_values[eval_name] = fmax_value
            else:
                print(f"{eval_name} / {key_name} not found")

        if len(fmax_values.keys()) == 0:
            print("No Fmax values found for this model.")
            continue
        else:
            full_strategy_name = target_type + " and " + model_name
            if not full_strategy_name in model_results:
                model_results[full_strategy_name] = []
            fmax_values["Parameters"] = str(param_comb)
            model_results[full_strategy_name].append(fmax_values)

for full_strategy_name, fmax_tests in model_results.items():
    # sort to get best CAFA result
    print(fmax_tests[0].keys())
    fmax_tests.sort(
        key=lambda x: x["NAN Fmax Macro"],
        reverse=True,
    )
    fmax_tests[0]["Number of Tests"] = len(fmax_tests)
    model_results[full_strategy_name] = fmax_tests[0]

lines = []
for model_full_name, fmax_values in model_results.items():
    new_line = {"Test Name": model_full_name}
    for key, val in fmax_values.items():
        if type(val) not in [str, list, dict, int]:
            val = round(val, 5)
        new_line[key] = val
    lines.append(new_line)
lines.sort(
    key=lambda x: round(x["CAFA Fmax Macro"], 2) + round(x["NAN Fmax Macro"], 2),
    reverse=True,
)
for l in lines:
    print(l)
df = pl.DataFrame(lines)
df.write_csv(output_path, separator="\t")
