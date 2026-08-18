import sys
import os
import json
from glob import glob

import pandas

test_dir = "outputs/pooling_exploration/"
result_jsons = glob(f"{test_dir}/*_test/results_eval.json")

lines = []
for json_file in result_jsons:
    with open(json_file, "r") as f:
        data = json.load(f)
        if "DEEPLOC - Sort Score" in data:
            data["Pooling Method"] = os.path.basename(os.path.dirname(json_file)).replace("_test", "").replace("_pooling", "").replace("_", "+").title()
            lines.append(data)

df = pandas.DataFrame(lines)
sort_score_cols = [p + " - Sort Score" for p in ["DEEPLOC", "MF", "CC", "BP"]]
df["mean"] = df[sort_score_cols].mean(axis=1)
df = df.sort_values(by="mean", ascending=False)
del df["mean"]
df.to_csv(f"{test_dir}/summary.csv", index=False)
df = df[["Pooling Method"] + sort_score_cols]
df.to_csv(f"{test_dir}/summary_short.csv", index=False)

