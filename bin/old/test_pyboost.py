import os

# Optional: set the device to run
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

os.makedirs("../data", exist_ok=True)
import numpy as np
import joblib
from sklearn.datasets import make_regression
import cupy as cp

# simple case - just one class is used
from py_boost import GradientBoosting
from py_boost.multioutput.sketching import *
from py_boost.gpu.losses.metrics import Metric, auc
from py_boost.gpu.losses import BCELoss

print("Loaded libs")


class BCEWithNaNLoss(BCELoss):

    def base_score(self, y_true):
        # Replace .mean with nanmean function to calc base score
        means = cp.clip(
            cp.nanmean(y_true, axis=0), self.clip_value, 1 - self.clip_value
        )
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


# And here is column-wise roc-auc metric ignoring NaNs
class NaNAucMetric(Metric):

    def __call__(self, y_true, y_pred, sample_weight=None):

        aucs = []
        mask = ~cp.isnan(y_true)

        for i in range(y_true.shape[1]):
            m = mask[:, i]
            w = None if sample_weight is None else sample_weight[:, 0][m]
            aucs.append(auc(y_true[:, i][m], y_pred[:, i][m], w))

        return np.mean(aucs)

    def compare(self, v0, v1):

        return v0 > v1


print("Loading dummy data")

X, y = make_regression(150000, 100, n_targets=10, random_state=42)
# binarize
y = (y > y.mean(axis=0)).astype(np.float32)
# add some NaNs
y[np.random.rand(150000, 10) > 0.5] = np.nan

X_test, y_test = X[:50000], y[:50000]
X, y = X[-50000:], y[-50000:]

print("Initializing model")
model = GradientBoosting(
    BCEWithNaNLoss(),
    NaNAucMetric(),
    lr=0.01,
    verbose=100,
    ntrees=1000,
    es=200,
    multioutput_sketch=RandomProjectionSketch(1),
)

print("Starting training")
model.fit(
    X,
    y,
    eval_sets=[
        {"X": X_test, "y": y_test},
    ],
)
