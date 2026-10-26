"""
Step 6a: XGBoost  (regression + Buy/Sell now + Buy/Sell 12 bars ahead), scored on VALIDATION
Run from inside the "Bitcoin MLBD" folder:   python run_xgboost.py
Needs bs_train.csv, bs_val.csv, bs_test.csv (Step 5) and common.py in the same folder.
Install first:  pip install xgboost scikit-learn
The test set is not touched here.
"""

import time
import warnings
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from common import (load_data, reg_metrics, clf_metrics, log_result,
                    log_baselines, show_table, SEED)

warnings.filterwarnings("ignore")

train, val, test, FEATURES = load_data()
X_train, X_val = train[FEATURES], val[FEATURES]

log_baselines(train, val)

COMMON = dict(n_estimators=1000, learning_rate=0.03, max_depth=3, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=5, early_stopping_rounds=50,
              random_state=SEED)


def top_features(model, n=6):
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    return ", ".join("{} ({:.2f})".format(k, v) for k, v in imp.head(n).items())


# ---------- 1. Regression: next-bar return ----------
t0 = time.time()
reg = XGBRegressor(eval_metric="rmse", **COMMON)
reg.fit(X_train, train["target_ret_clip"], eval_set=[(X_val, val["target_ret_clip"])], verbose=False)
secs = time.time() - t0
m = reg_metrics(val["target_ret_clip"], reg.predict(X_val))
log_result("XGBoost", "regression", "validation", m, secs, "{} trees".format(reg.best_iteration + 1))
print("Regression done: {} trees, {:.1f}s".format(reg.best_iteration + 1, secs))
print("  Top features:", top_features(reg))

# ---------- 2. Classification: Buy/Sell now ----------
t0 = time.time()
clf = XGBClassifier(eval_metric="logloss", **COMMON)
clf.fit(X_train, train["target_class"], eval_set=[(X_val, val["target_class"])], verbose=False)
secs = time.time() - t0
m = clf_metrics(val["target_class"], clf.predict_proba(X_val)[:, 1])
log_result("XGBoost", "class_now", "validation", m, secs, "{} trees".format(clf.best_iteration + 1))
print("Classification (now) done: {} trees, {:.1f}s".format(clf.best_iteration + 1, secs))
print("  Top features:", top_features(clf))

# ---------- 3. Classification: Buy/Sell 12 bars ahead (extra) ----------
t0 = time.time()
clf12 = XGBClassifier(eval_metric="logloss", **COMMON)
clf12.fit(X_train, train["target_class_12"], eval_set=[(X_val, val["target_class_12"])], verbose=False)
secs = time.time() - t0
m = clf_metrics(val["target_class_12"], clf12.predict_proba(X_val)[:, 1])
log_result("XGBoost", "class_12", "validation", m, secs, "{} trees".format(clf12.best_iteration + 1))
print("Classification (12 ahead) done: {} trees, {:.1f}s".format(clf12.best_iteration + 1, secs))
print("  Top features:", top_features(clf12))

print("\n" + "=" * 70)
print("VALIDATION RESULTS SO FAR (saved in results_all.csv)")
print("=" * 70)
show_table("validation")
print("\nHow to read it:")
print("  Regression: lower RMSE/MAE is better, R2 above 0 beats the 'always predict the average' baseline.")
print("  Classification: compare Accuracy, BalancedAcc, F1 and AUC against the Baseline row.")
