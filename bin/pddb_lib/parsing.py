import polars as pl
import pandas as pd
import numpy as np

def reorder_parquet(df: pl.DataFrame, ids_sorted: list):
    order_expr = pl.lit(ids_sorted).to_series()
    df = df.sort_by([order_expr])
    return df

def load_data(targets_path: str, feature_descs, ids_subset: set=None):

    ids_in_all = set()
    ids_in_all.update(ids_subset)

    for parquet_path in [x[0] for x in feature_descs] + [targets_path]:
        #Load just the id column
        parquet_ids = set(pl.scan_parquet(parquet_path).select("id").collect()["id"].to_list())
        ids_in_all = ids_in_all.intersection(parquet_ids)
    
    ids_to_use_list = list(ids_in_all)

    targets_df = pl.scan_parquet(targets_path).filter(pl.col("id").is_in(ids_to_use_list)).collect()
    targets_df = reorder_parquet(targets_df, ids_to_use_list)
        
    target_ids = targets_df["id"].to_list()
    id_to_index = {id: i for i, id in enumerate(target_ids)}
    n_proteins = len(target_ids)

    x_cols = []
    for parquet_path, col_names, embname in feature_descs:
        x = [np.nan for _ in range(n_proteins)]
        emb_df = (
            pl.scan_parquet(parquet_path)
            .filter(pl.col("id").is_in(ids_to_use_list))
            .collect()
        )
        emb_df = reorder_parquet(emb_df, ids_to_use_list)
        for row in emb_df.iter_rows(named=True):
            idx = id_to_index[row["id"]]
            x[idx] = np.concatenate([row[c] for c in col_names])
        x = np.asarray(x)

        x_cols.append(x)
    all_features_x = np.concatenate(x_cols, axis=1)
    df_dict = {
        "id": target_ids,
        "X": all_features_x,
    }
    for col_name in targets_df.columns:
        if "y_" in col_name:
            df_dict[col_name] = targets_df[col_name].to_numpy()
    df = pl.DataFrame(df_dict)

    return df

def load_data_optimized(targets_path: str, feature_descs: list, ids_subset: set = None, 
        y_to_use: list = None) -> pl.DataFrame:
    # 1. Initialize the base LazyFrame
    if y_to_use != None:
        #Load only the specified columns
        base_lf = pl.scan_parquet(targets_path).select(["id"] + y_to_use)
    else:
        #Load all y columns
        base_lf = pl.scan_parquet(targets_path)
    
    # Filter targets immediately if a subset is provided to minimize memory usage
    if ids_subset is not None:
        base_lf = base_lf.filter(pl.col("id").is_in(list(ids_subset)))

    y_cols = [c for c in base_lf.collect_schema().names() if "y_" in c]
    all_feature_cols = []
    print(base_lf)

    # 2. Iteratively Inner Join feature datasets
    # This natively finds the exact intersection of all IDs across all files 
    # AND perfectly aligns the rows, eliminating the need for sorting.
    for fdesc_raw in feature_descs:
        parquet_path = fdesc_raw.split(':')[0]
        col_names = fdesc_raw.split(':')[1].split(',')
        # Select only 'id' and the necessary features to avoid loading unused columns
        feat_lf = pl.scan_parquet(parquet_path).select(["id"] + col_names)
        
        # Inner join automatically handles matching and filtering
        base_lf = base_lf.join(feat_lf, on="id", how="inner")
        all_feature_cols.extend(col_names)
        print(base_lf)

    # 3. Trigger the Polars execution graph once
    df = base_lf.collect()
    print(df)

    # 4. Extract target arrays
    df_dict = {
        "id": df["id"].to_list()
    }
    for col_name in y_cols:
        df_dict[col_name] = df[col_name].to_numpy()

    # 5. Extract and concatenate X features via vectorized NumPy
    x_arrays = []
    for col in all_feature_cols:
        series = df[col]
        # Handle embedding/array columns natively
        if isinstance(series.dtype, (pl.List, pl.Array)):
            # stack(to_list()) is the fastest way to get a 2D numpy matrix from list columns
            arr = np.stack(series.to_list()) 
        else:
            # Handle scalar columns by reshaping them to (N, 1) for horizontal concatenation
            arr = series.to_numpy().reshape(-1, 1)
        x_arrays.append(arr)
    
    # Concatenate all matrices horizontally
    df_dict["X"] = list(np.concatenate(x_arrays, axis=1))
    new_df = pl.DataFrame(df_dict)
    print(new_df)

    # Return as a new Polars DataFrame
    return new_df

def smart_str_parsing(val: str):
    if type(val) in [np.int64, np.int32]:
        return int(val)
    elif type(val) in [np.float64, np.float32]:
        return float(val)
    elif type(val) in [np.bool_]:
        return bool(val)
    elif "." in val and val.replace('.', '').isdigit():
        return float(val)
    elif val.isdigit():
        return int(val)
    elif val.lower() in ['true', 'false']:
        return val.lower() == "true"
    else:
        print(val, 'is a string')
        return val