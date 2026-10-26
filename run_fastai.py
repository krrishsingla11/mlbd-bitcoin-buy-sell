"""
Step 6c: PyTorch + fastai (regression + Buy/Sell now + Buy/Sell 12 bars ahead), on VALIDATION
Run from inside the "Bitcoin MLBD" folder:   python run_fastai.py
Needs bs_train.csv, bs_val.csv, bs_test.csv (Step 5) and common.py in the same folder.
Install first:  pip install fastai
The test set is not touched here.
"""

import io
import time
import warnings
from contextlib import redirect_stdout
import numpy as np
import pandas as pd
from fastai.tabular.all import (TabularPandas, Normalize, CategoryBlock, RegressionBlock,
                                tabular_learner, tabular_config, EarlyStoppingCallback,
                                SaveModelCallback, set_seed)
from common import (load_data, reg_metrics, clf_metrics, log_result,
                    log_baselines, show_table, SEED)

warnings.filterwarnings("ignore")

train, val, test, FEATURES = load_data()
log_baselines(train, val)

n_train = len(train)
splits = (list(range(n_train)), list(range(n_train, n_train + len(val))))   # train rows, then validation rows


def run(task, target_col, y_scaled=None):
    """Train one fastai tabular model and return predictions for the validation set."""
    df = pd.concat([train[FEATURES], val[FEATURES]]).reset_index(drop=True)
    if task == "regression":
        df["y"] = np.concatenate([y_scaled[0], y_scaled[1]]).astype("float32")
        block = RegressionBlock()
    else:
        df["y"] = np.concatenate([train[target_col].values, val[target_col].values]).astype(int)
        block = CategoryBlock()

    # Normalize learns its mean/spread from the TRAIN rows only
    to = TabularPandas(df, procs=[Normalize], cat_names=[], cont_names=FEATURES,
                       y_names="y", y_block=block, splits=splits)
    dls = to.dataloaders(bs=64, num_workers=0)       # num_workers=0 avoids Windows multiprocessing issues

    set_seed(SEED, reproducible=True)
    learn = tabular_learner(dls, layers=[64, 32], config=tabular_config(ps=[0.3, 0.3]))
    n_weights = sum(p.numel() for p in learn.model.parameters())

    t0 = time.time()
    cbs = [EarlyStoppingCallback(monitor="valid_loss", patience=15),
           SaveModelCallback(monitor="valid_loss", fname="best_" + task)]   # keeps the best epoch
    with learn.no_bar(), learn.no_logging(), redirect_stdout(io.StringIO()):   # keep the screen clean
        learn.fit(200, lr=1e-3, wd=1e-4, cbs=cbs)
    secs = time.time() - t0

    valid_losses = [v[1] for v in learn.recorder.values]
    best_epoch = int(np.argmin(valid_losses)) + 1

    dl = learn.dls.test_dl(val[FEATURES])
    with learn.no_bar(), redirect_stdout(io.StringIO()):
        preds, _ = learn.get_preds(dl=dl)
    return preds.numpy(), secs, best_epoch, n_weights


# ---------- 1. Regression: next-bar return ----------
y_mean, y_std = train["target_ret_clip"].mean(), train["target_ret_clip"].std()
y_tr = ((train["target_ret_clip"] - y_mean) / y_std).values
y_va = ((val["target_ret_clip"] - y_mean) / y_std).values
preds, secs, best, n_w = run("regression", None, (y_tr, y_va))
pred = preds.ravel() * y_std + y_mean
log_result("fastai", "regression", "validation", reg_metrics(val["target_ret_clip"], pred), secs,
           "{} epochs, {} weights".format(best, n_w))
print("Regression done: best at epoch {}, {:.1f}s".format(best, secs))

# ---------- 2 and 3. Classification ----------
for task, col, label in [("class_now", "target_class", "Buy/Sell now"),
                         ("class_12", "target_class_12", "Buy/Sell 12 bars ahead")]:
    preds, secs, best, n_w = run(task, col)
    prob = preds[:, 1]                                # probability of class 1 = Buy
    log_result("fastai", task, "validation", clf_metrics(val[col], prob), secs,
               "{} epochs, {} weights".format(best, n_w))
    print("Classification ({}) done: best at epoch {}, {:.1f}s".format(label, best, secs))

print("\n" + "=" * 70)
print("VALIDATION RESULTS SO FAR (saved in results_all.csv)")
print("=" * 70)
show_table("validation")
