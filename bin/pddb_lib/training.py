import numpy as np
from py_boost import GradientBoosting

from pddb_lib.fuzzy_ml import BCEWithNaNLoss, BCEwithNaNMetric
from pddb_lib.custom_statistics import fmax, fmax_dual, macro_fmax_dual

def add_random_false_values(train_y, target_min_zeros=0.12, zero_val=0.0):
    # x != NaN and x < 0.5 = negative evi = zero
    zeros_mask = (~np.isnan(train_y)) & (train_y < 0.5)

    n_cells_total = train_y.size
    n_zeros_min = int(n_cells_total * target_min_zeros)
    n_zeros_current = np.sum(zeros_mask)

    if n_zeros_current < n_zeros_min:
        n_zeros_to_add = n_zeros_min - n_zeros_current

        # Encontra as coordenadas (flattened) de todos os NaNs disponíveis
        nan_indices = np.where(np.isnan(train_y).flatten())[0]

        # Garante que não vamos tentar amostrar mais NaNs do que o disponível
        n_zeros_to_add = min(n_zeros_to_add, len(nan_indices))

        if n_zeros_to_add > 0:
            # 2. Seleção exata e aleatória sem repetição (garante o número preciso)
            chosen_nan_indices = np.random.choice(
                nan_indices, size=n_zeros_to_add, replace=False
            )

            # Desachata os índices para o formato da matriz original e substitui
            train_y.ravel()[chosen_nan_indices] = zero_val
        return train_y, True
    else:
        # No need to add more falses
        return train_y, False

def eval_param_comb(params_dict, train_x, test_x, train_y, test_y, mask_nan=True):
    try:
        if mask_nan:
            model = GradientBoosting(
                # "bce",
                BCEWithNaNLoss(),
                BCEwithNaNMetric(),
                ntrees=params_dict["ntrees"],
                lr=params_dict["lr"],
                min_gain_to_split=params_dict["min_gain_to_split"],
                es=params_dict["es"],
                lambda_l2=params_dict["lambda_l2"],
                gd_steps=params_dict["gd_steps"],
                subsample=params_dict["subsample"],
                colsample=params_dict["colsample"],
                min_data_in_leaf=params_dict["min_data_in_leaf"],
                use_hess=params_dict["use_hess"],
                max_bin=params_dict["max_bin"],
                max_depth=params_dict["max_depth"],
            )
        else:
            model = GradientBoosting(
                "bce",
                ntrees=params_dict["ntrees"],
                lr=params_dict["lr"],
                min_gain_to_split=params_dict["min_gain_to_split"],
                es=params_dict["es"],
                lambda_l2=params_dict["lambda_l2"],
                gd_steps=params_dict["gd_steps"],
                subsample=params_dict["subsample"],
                colsample=params_dict["colsample"],
                min_data_in_leaf=params_dict["min_data_in_leaf"],
                use_hess=params_dict["use_hess"],
                max_bin=params_dict["max_bin"],
                max_depth=params_dict["max_depth"],
            )

        model.fit(train_x, train_y, eval_sets=[{"X": test_x, "y": test_y}])

        y_pred = model.predict(test_x)

        # fmax_fuzzy, fmax_cafa = fmax_dual(test_y, y_pred)
        fmax_fuzzy, _, fmax_fuzzy_all, _ = macro_fmax_dual(test_y, y_pred)
        print(f"All FMAX values:")
        print(f"Fuzzy: {fmax_fuzzy_all}")

        del model
        import gc

        gc.collect()

        return fmax_fuzzy, fmax_fuzzy_all, True
    except Exception as e:
        print(f"Error in param combination {params_dict}: {e}")
        return -1, -1, False


def train_param_comb(params_dict, train_x, test_x, train_y, test_y):
    try:
        model = GradientBoosting(
            # "bce",
            BCEWithNaNLoss(),
            BCEwithNaNMetric(),
            ntrees=params_dict["ntrees"],
            lr=params_dict["lr"],
            min_gain_to_split=params_dict["min_gain_to_split"],
            es=params_dict["es"],
            lambda_l2=params_dict["lambda_l2"],
            gd_steps=params_dict["gd_steps"],
            subsample=params_dict["subsample"],
            colsample=params_dict["colsample"],
            min_data_in_leaf=params_dict["min_data_in_leaf"],
            use_hess=params_dict["use_hess"],
            max_bin=params_dict["max_bin"],
            max_depth=params_dict["max_depth"],
        )

        model.fit(train_x, train_y, eval_sets=[{"X": test_x, "y": test_y}])

        y_pred_train = model.predict(train_x)
        y_pred_test = model.predict(test_x)

        del model
        import gc

        gc.collect()

        return y_pred_test, y_pred_train, True
    except Exception as e:
        print(f"Error in param combination {params_dict}: {e}")
        return None, None, False

def train_and_pred(train_x, train_y, test_x, test_y, params_dict):
    model = GradientBoosting(
        # "bce",
        BCEWithNaNLoss(),
        BCEwithNaNMetric(),
        ntrees=params_dict["ntrees"],
        lr=params_dict["lr"],
        min_gain_to_split=params_dict["min_gain_to_split"],
        es=params_dict["es"],
        lambda_l2=params_dict["lambda_l2"],
        gd_steps=params_dict["gd_steps"],
        subsample=params_dict["subsample"],
        colsample=params_dict["colsample"],
        min_data_in_leaf=params_dict["min_data_in_leaf"],
        use_hess=True,
        max_bin=params_dict["max_bin"],
        max_depth=params_dict["max_depth"],
    )

    model.fit(train_x, train_y, eval_sets=[{"X": test_x, "y": test_y}])

    y_pred_test = model.predict(test_x)

    del model
    import gc

    gc.collect()

    return y_pred_test

def train_and_pred_failsafe(train_x, train_y, test_x, test_y, params_dict):
    try:
        return train_and_pred(train_x, train_y, test_x, test_y, params_dict)
    except Exception as e:
        print(f"Error with param combination {params_dict}: {e}")
        return None
