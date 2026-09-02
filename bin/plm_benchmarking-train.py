import sys
import os
import json

import numpy as np

from pddb_lib.training import train_and_pred

if __name__ == "__main__":

    model_dir = sys.argv[1]
    ont = sys.argv[2]

    test_ids_path = os.path.join(model_dir, f"{ont}-test_ids.txt")
    train_x_path = os.path.join(model_dir, f"{ont}-train_x.npy")
    train_y_path = os.path.join(model_dir, f"{ont}-train_y.npy")
    test_x_path = os.path.join(model_dir, f"{ont}-test_x.npy")
    test_y_path = os.path.join(model_dir, f"{ont}-test_y.npy")
    y_pred_path = os.path.join(model_dir, f"{ont}-y_pred.npy")
    log_path = os.path.join(model_dir, f"{ont}-train.log")
    params_path = os.path.join(model_dir, "params.json")

    test_ids = [line.strip() for line in open(test_ids_path)]
    train_x = np.load(train_x_path)
    train_y = np.load(train_y_path)
    test_x = np.load(test_x_path)
    test_y = np.load(test_y_path)
    metaparameters = json.load(open(params_path))
    
    y_pred = train_and_pred(train_x, train_y, test_x, test_y, 
        metaparameters, True)
    np.save(y_pred_path, y_pred)