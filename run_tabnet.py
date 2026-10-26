"""
Step 6d: PyTorch + TabNet (regression + Buy/Sell now + Buy/Sell 12 bars ahead), on VALIDATION
Run from inside the "Bitcoin MLBD" folder:   python run_tabnet.py
Needs bs_train.csv, bs_val.csv, bs_test.csv (Step 5) and common.py in the same folder.
Install first:  pip install pytorch-tabnet
The test set is not touched here.
"""

import time
import warnings
import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from common import (load_data, scale_with_train, reg_metrics, clf_metrics,
                    log_result, log_baselines, show_table, SEED)

warnings.filterwarnings("ignore")

train, val, test, FEATURES = load_data()
log_baselines(train, val)

X_train, X_val = scale_with_train(train[FEATURES], val[FEATURES])
X_train, X_val = X_train.values.astype("float32"), X_val.values.astype("float32")

# Small TabNet: 3 decision steps, 8 units each (keeps it comparable with the other networks)
PARAMS = dict(n_d=8, n_a=8, n_steps=3, gamma=1.3, n_independent=1, n_shared=1,
              optimizer_params=dict(lr=1e-2), seed=SEED, verbose=0)
FIT = dict(max_epochs=200, patience=15, batch_size=256, virtual_batch_size=64)


def n_weights(model):
    return sum(p.numel() for p in model.network.parameters())


def top_features(model, n=6):
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    return ", ".join("{} ({:.2f})".format(k, v) for k, v in imp.head(n).items())


# ---------- 1. Regression: next-bar return ----------
torch.manual_seed(SEED)
y_mean, y_std = train["target_ret_clip"].mean(), train["target_ret_clip"].std()
y_tr = ((train["target_ret_clip"] - y_mean) / y_std).values.astype("float32").reshape(-1, 1)
y_va = ((val["target_ret_clip"] - y_mean) / y_std).values.astype("float32").reshape(-1, 1)

t0 = time.time()
reg = TabNetRegressor(**PARAMS)
reg.fit(X_train, y_tr, eval_set=[(X_val, y_va)], eval_metric=["rmse"], **FIT)
secs = time.time() - t0
pred = reg.predict(X_val).ravel() * y_std + y_mean
log_result("TabNet", "regression", "validation", reg_metrics(val["target_ret_clip"], pred), secs,
           "{} epochs, {} weights".format(reg.best_epoch + 1, n_weights(reg)))
print("Regression done: best at epoch {}, {:.1f}s".format(reg.best_epoch + 1, secs))
print("  Top features:", top_features(reg))

# ---------- 2 and 3. Classification ----------
for task, col, label in [("class_now", "target_class", "Buy/Sell now"),
                         ("class_12", "target_class_12", "Buy/Sell 12 bars ahead")]:
    torch.manual_seed(SEED)
    t0 = time.time()
    clf = TabNetClassifier(**PARAMS)
    clf.fit(X_train, train[col].values, eval_set=[(X_val, val[col].values)],
            eval_metric=["logloss"], **FIT)
    secs = time.time() - t0
    prob = clf.predict_proba(X_val)[:, 1]
    log_result("TabNet", task, "validation", clf_metrics(val[col], prob), secs,
               "{} epochs, {} weights".format(clf.best_epoch + 1, n_weights(clf)))
    print("Classification ({}) done: best at epoch {}, {:.1f}s".format(label, clf.best_epoch + 1, secs))
    print("  Top features:", top_features(clf))

print("\n" + "=" * 70)
print("VALIDATION RESULTS SO FAR (saved in results_all.csv)")
print("=" * 70)
show_table("validation")
