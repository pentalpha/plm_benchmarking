from py_boost import GradientBoosting
from fuzzy_ml import BCEWithNaNLoss, BCEwithNaNMetric

from custom_statistics import fmax, fmax_dual, macro_fmax_dual


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
