import sys
import os
import json
import random

import numpy as np
from pddb_lib.sample_metaparameters import generate_for_genelist, gdbt_params_list

n_combinations = int(sys.argv[1])
test_dir = "outputs/plm_benchmarking/"
combinations_path = f"{test_dir}/parameter_options.json"
predefs_json_path = "outputs/evi_exploration_results.json"

existing_param_combs = []
if os.path.exists(combinations_path):
    with open(combinations_path, "r") as f:
        try:
            existing_param_combs = json.load(f)
        except json.JSONDecodeError:
            existing_param_combs = []
else:
    predefs = json.load(open(predefs_json_path, 'r'))["soft"]
    existing_param_combs = []
    for ont, params in predefs.items():
        params_sub = {k: v for k, v in params.items() if k in gdbt_params_list}
        existing_param_combs.append(params_sub)

#set python and numpy seeds
random.seed(1337)
np.random.seed(1337)

while len(existing_param_combs) < n_combinations:
    print(f"Existing {len(existing_param_combs)}/{n_combinations} parameter combinations. Generating more...")
    n_new_combs = n_combinations - len(existing_param_combs)
    existing_param_combs_str = {json.dumps(c) for c in existing_param_combs}
    new_combs = generate_for_genelist(n_new_combs*2, gdbt_params_list, try_more=True)
    new_combs = [{k: v for k, v in zip(gdbt_params_list, c)} for c in new_combs]
    
    for key in gdbt_params_list:
        for c in new_combs:
            if type(c[key]) == np.float64:
                c[key] = float(c[key])
            elif type(c[key]) == np.int64:
                c[key] = int(c[key])
            elif type(c[key]) == np.bool_:
                c[key] = bool(c[key])
    
    new_combs = [json.dumps(c) for c in new_combs]
    
    actually_new = set()
    for comb in new_combs:
        if comb not in existing_param_combs_str:
            actually_new.add(comb)

    print(f"Found {len(actually_new)} new unique parameter combinations.")
    
    if len(actually_new) > n_new_combs:
        new_combs = list(new_combs)
        indexes = list([i for i in range(len(new_combs))])
        indexes = random.sample(indexes, n_new_combs)
        actually_new = [new_combs[idx] for idx in indexes]
        print(f"Selected {len(actually_new)} parameter combinations.")
    else:
        actually_new = list(actually_new)
    
    actually_new2 = [json.loads(c) for c in actually_new]
    
    existing_param_combs.extend(actually_new2)

for i, comb in enumerate(existing_param_combs):
    print(i, f"{len(comb)} values:", comb)

with open(combinations_path, "w") as f:
    json.dump(existing_param_combs, f, indent=4, ensure_ascii=False)

