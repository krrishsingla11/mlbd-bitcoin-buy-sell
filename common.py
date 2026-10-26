"""
Shared helpers for all framework scripts (XGBoost, Keras, fastai, TabNet, Lightning).
Keep this file in the same folder as the run_*.py scripts. You do not run it directly.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "results_all.csv")

TARGETS = ["target_class", "target_class_12", "target_ret", "target_ret_clip"]
SEED = 42


def load_data():
    """Returns train, val, test tables and the list of input feature names."""
    train = pd.read_csv(os.path.join(HERE, "bs_train.csv"), index_col="Date", parse_dates=True)
    val = pd.read_csv(os.path.join(HERE, "bs_val.csv"), index_col="Date", parse_dates=True)
    test = pd.read_csv(os.path.join(HERE, "bs_test.csv"), index_col="Date", parse_dates=True)
    features = [c for c in train.columns if c not in TARGETS]
    return train, val, test, features


def scale_with_train(train_X, *others):
    """Standardise features using the TRAIN set's mean and spread only (for neural networks)."""
    mean, std = train_X.mean(), train_X.std().replace(0, 1)
    out = [(train_X - mean) / std]
    out += [(x - mean) / std for x in others]
    return out


def reg_metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def clf_metrics(y_true, prob, threshold=0.5):
    pred = (np.asarray(prob) >= threshold).astype(int)
    try:
        auc = float(roc_auc_score(y_true, prob))
    except ValueError:
        auc = float("nan")
    return {
        "Accuracy": float(accuracy_score(y_true, pred)),
        "BalancedAcc": float(balanced_accuracy_score(y_true, pred)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "AUC": auc,
    }


def log_result(framework, task, split, metrics, seconds=None, notes=""):
    """Add (or replace) one row in results_all.csv so we can compare frameworks later."""
    row = {"Framework": framework, "Task": task, "Split": split, "Seconds": seconds, "Notes": notes}
    row.update(metrics)
    new = pd.DataFrame([row])
    if os.path.exists(RESULTS_PATH):
        old = pd.read_csv(RESULTS_PATH)
        keep = ~((old["Framework"] == framework) & (old["Task"] == task) & (old["Split"] == split))
        new = pd.concat([old[keep], new], ignore_index=True)
    new.to_csv(RESULTS_PATH, index=False)


def log_baselines(train, eval_df, split="validation"):
    """Simple 'do nothing smart' scores that every model has to beat."""
    # Regression: always predict the average train return
    pred = np.full(len(eval_df), train["target_ret_clip"].mean())
    log_result("Baseline", "regression", split, reg_metrics(eval_df["target_ret_clip"], pred),
               notes="always predict the train average")
    if "target_ret" in eval_df.columns and split == "test":
        log_result("Baseline", "regression_raw", split, reg_metrics(eval_df["target_ret"], pred),
                   notes="always predict the train average")

    # Classification now: always predict the most common train class
    majority = int(train["target_class"].mean() >= 0.5)
    log_result("Baseline", "class_now", split,
               clf_metrics(eval_df["target_class"], np.full(len(eval_df), float(majority))),
               notes="always predict " + ("Buy" if majority else "Sell"))

    # Classification 12 ahead: repeat the current signal
    log_result("Baseline", "class_12", split,
               clf_metrics(eval_df["target_class_12"], eval_df["target_class"].astype(float)),
               notes="repeat the current signal")


def show_table(split="validation"):
    """Print everything logged so far for one split."""
    if not os.path.exists(RESULTS_PATH):
        return
    r = pd.read_csv(RESULTS_PATH)
    r = r[r["Split"] == split]
    pd.set_option("display.width", 220)
    for task in ["regression", "regression_raw", "class_now", "class_12"]:
        part = r[r["Task"] == task].dropna(axis=1, how="all").drop(columns=["Task", "Split"])
        if len(part):
            print("\n[{}]".format(task))
            print(part.round(4).to_string(index=False))
