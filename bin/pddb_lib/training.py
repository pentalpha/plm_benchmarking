from copy import copy
from random import sample
from py_boost import Callback
from py_boost.gpu.losses import BCELoss, BCEMetric
from py_boost import GradientBoosting
from py_boost.multioutput.sketching import RandomSamplingSketch
import numpy as np
import cupy as cp
import networkx as nx
import obonet

from pddb_lib.custom_statistics import fmax, fmax_dual, macro_fmax_dual

class BCEWithNaNLoss(BCELoss):

    def base_score(self, y_true):
        # Replace .mean with nanmean function to calc base score
        means = cp.nanmean(y_true, axis=0)
        means = cp.where(cp.isnan(means), 0, means)
        means = cp.clip(means, self.clip_value, 1 - self.clip_value)

        return cp.log(means / (1 - means))

    def get_grad_hess(self, y_true, y_pred):
        # first, get nan mask for y_true
        mask = cp.isnan(y_true)
        # then, compute loss with any values at nan places just to prevent the exception
        grad, hess = super().get_grad_hess(cp.where(mask, 0, y_true), y_pred)
        # invert mask
        mask = (~mask).astype(cp.float32)
        # multiply grad and hess on inverted mask
        # now grad and hess eq. 0 on NaN points
        # that actually means that prediction on that place should not be updated
        grad = grad * mask
        hess = hess * mask

        return grad, hess


class BCEwithNaNMetric(BCEMetric):

    def __call__(self, y_true, y_pred, sample_weight=None):
        mask = ~cp.isnan(y_true)

        err = super().error(cp.where(mask, y_true, 0), y_pred)
        err = err * mask

        if sample_weight is not None:
            err = err * sample_weight
            mask = mask * sample_weight

        return float(err.sum() / mask.sum())


class WarmStart(Callback):

    def __init__(self, model):
        model.to_cpu()
        self.model = copy(model)
        self.model.postprocess_fn = lambda x: x

    def before_train(self, build_info):
        build_info["model"].base_score = cp.asarray(self.model.base_score)

        train = build_info["data"]["train"]
        train["ensemble"] = cp.asarray(self.model.predict(train["features_cpu"]))

        valid = build_info["data"]["valid"]
        valid["ensemble"] = [
            cp.asarray(self.model.predict(x)) for x in valid["features_cpu"]
        ]

        self.model.to_cpu()

        return

    def after_train(self, build_info):
        build_info["model"].models = self.model.models + build_info["model"].models
        # update the actual iteration
        build_info["num_iter"] = build_info["num_iter"] + len(self.model.models)
        # update the actual best round
        early_stop = build_info["model"].callbacks.callbacks[-1]
        early_stop.best_round = early_stop.best_round + len(self.model.models)

        # not to store old trees multiple times
        self.model = None

        return

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

def reduce_train_negatives_to(train_x, train_y, target_ratio, use_nan=True):
    # 1. Verifica no eixo 1 (colunas/targets) se TODOS os valores da linha são NaN ou 0
    print("Train_x shape: ", train_x.shape)
    print("Train_y shape: ", train_y.shape)
    if use_nan:
        is_zero = np.isnan(train_y).all(axis=1)
    else:
        is_zero = (train_y == 0).all(axis=1)
    
    # 2. Pega os índices 1D das linhas
    negatives = np.where(is_zero)[0]
    with_info = np.where(~is_zero)[0]

    n_negatives = len(negatives)
    n_with_info = len(with_info)
    
    max_negatives = round(n_with_info * (target_ratio / (1 - target_ratio)))

    if n_negatives > max_negatives:
        print(f"Reducing negatives from {n_negatives} to {max_negatives} to match {target_ratio} ratio of positives.")
        
        # 3. Faz o sample APENAS em cima dos índices negativos
        sampled_negatives = np.random.choice(negatives, size=max_negatives, replace=False)
        
        # 4. Mantém TODAS as linhas que tem informação + a amostra de negativos
        indexes_to_keep = np.concatenate((sampled_negatives, with_info))
        
        # Mistura os índices para não agrupar todos os negativos de um lado (boa prática pro GBDT)
        np.random.shuffle(indexes_to_keep)
        
        train_x = train_x[indexes_to_keep]
        train_y = train_y[indexes_to_keep]
    else:
        print(f"No need to reduce negatives. {n_negatives} <= {max_negatives}.")
        
    return train_x, train_y
    


def train_and_pred(train_x, train_y, test_x, test_y, params_dict, has_nan):
    ncols = train_y.shape[1]
    sketch_perc = 0.2
    sketch_size = round(ncols * sketch_perc)
    sketch_size = max(1, sketch_size)
    print(f"Sketching {sketch_size} out of {ncols} columns...")

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
        multioutput_sketch=RandomSamplingSketch(sketch_size)
    )

    print(params_dict)
    print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)
    print("train_x: ", train_x)
    print("test_x: ", test_x)
    print("train_y: ", train_y)
    print("test_y: ", test_y)

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
