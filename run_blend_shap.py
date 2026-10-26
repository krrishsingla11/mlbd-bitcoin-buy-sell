"""
Step 7: Blend XGBoost + Keras, then explain XGBoost with SHAP  (VALIDATION only)
Run from inside the "Bitcoin MLBD" folder:   python run_blend_shap.py
Needs bs_train.csv, bs_val.csv, bs_test.csv (Step 5) and common.py in the same folder.
Install first:  pip install shap matplotlib
Saves charts into the "plots" folder. The test set is not touched here.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras
from keras import layers, regularizers
import shap
from xgboost import XGBRegressor, XGBClassifier
from common import (HERE, load_data, scale_with_train, reg_metrics, clf_metrics,
                    log_result, log_baselines, show_table, SEED)

warnings.filterwarnings("ignore")
PLOTS = os.path.join(HERE, "plots")
os.makedirs(PLOTS, exist_ok=True)

train, val, test, FEATURES = load_data()
log_baselines(train, val)
X_train_raw, X_val_raw = train[FEATURES], val[FEATURES]
Xs_train, Xs_val = scale_with_train(X_train_raw, X_val_raw)
Xs_train, Xs_val = Xs_train.values.astype("float32"), Xs_val.values.astype("float32")

XGB_PARAMS = dict(n_estimators=1000, learning_rate=0.03, max_depth=3, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=5, early_stopping_rounds=50,
                  random_state=SEED)

TASKS = [  # (task name, target column, kind, nice label)
    ("regression", "target_ret_clip", "reg", "Regression (next-bar return)"),
    ("class_now", "target_class", "clf", "Buy/Sell now"),
    ("class_12", "target_class_12", "clf", "Buy/Sell 12 bars ahead"),
]


def keras_model(kind):
    m = keras.Sequential([
        keras.Input(shape=(len(FEATURES),)),
        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        layers.Dense(1, activation="linear" if kind == "reg" else "sigmoid"),
    ])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse" if kind == "reg" else "binary_crossentropy")
    return m


def metrics_for(kind, y_true, pred):
    return reg_metrics(y_true, pred) if kind == "reg" else clf_metrics(y_true, pred)


xgb_models = {}
rows_by_task = {}

for task, col, kind, label in TASKS:
    t0 = time.time()
    y_tr, y_va = train[col], val[col]

    # --- XGBoost ---
    if kind == "reg":
        xgb = XGBRegressor(eval_metric="rmse", **XGB_PARAMS)
    else:
        xgb = XGBClassifier(eval_metric="logloss", **XGB_PARAMS)
    xgb.fit(X_train_raw, y_tr, eval_set=[(X_val_raw, y_va)], verbose=False)
    xgb_pred = xgb.predict(X_val_raw) if kind == "reg" else xgb.predict_proba(X_val_raw)[:, 1]
    xgb_models[task] = xgb

    # --- Keras (target scaled for regression) ---
    keras.utils.set_random_seed(SEED)
    net = keras_model(kind)
    if kind == "reg":
        mu, sd = y_tr.mean(), y_tr.std()
        yt, yv = ((y_tr - mu) / sd).values, ((y_va - mu) / sd).values
    else:
        yt, yv = y_tr.values.astype("float32"), y_va.values.astype("float32")
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    net.fit(Xs_train, yt, validation_data=(Xs_val, yv), epochs=200, batch_size=64,
            callbacks=[stop], verbose=0)
    keras_pred = net.predict(Xs_val, verbose=0).ravel()
    if kind == "reg":
        keras_pred = keras_pred * sd + mu

    # --- Blend: simple average of the two predictions ---
    blend_pred = (xgb_pred + keras_pred) / 2
    secs = time.time() - t0

    rows = pd.DataFrame({
        "XGBoost": metrics_for(kind, y_va, xgb_pred),
        "Keras": metrics_for(kind, y_va, keras_pred),
        "Blend (average)": metrics_for(kind, y_va, blend_pred),
    }).T
    rows_by_task[task] = rows
    log_result("Blend (XGB+Keras)", task, "validation", metrics_for(kind, y_va, blend_pred), secs,
               "average of XGBoost and Keras")

print("=" * 70)
print("BLENDING: XGBoost + Keras averaged (validation set)")
print("=" * 70)
for task, col, kind, label in TASKS:
    print("\n[{}]".format(label))
    print(rows_by_task[task].round(4).to_string())

# ---------------- SHAP ----------------
print("\n" + "=" * 70)
print("INTERPRETABILITY: XGBoost built-in importance vs SHAP (validation set)")
print("=" * 70)


def save(name):
    plt.savefig(os.path.join(PLOTS, name), dpi=150, bbox_inches="tight")
    plt.close("all")


for task, col, kind, label in TASKS:
    model = xgb_models[task]
    explainer = shap.TreeExplainer(model)
    sv = explainer(X_val_raw)                       # SHAP values for every validation row

    builtin = pd.Series(model.feature_importances_, index=FEATURES)
    mean_abs = pd.Series(np.abs(sv.values).mean(axis=0), index=FEATURES)
    spearman = builtin.rank().corr(mean_abs.rank())

    print("\n[{}]".format(label))
    print("  XGBoost built-in top 5:", ", ".join(builtin.sort_values(ascending=False).head(5).index))
    print("  SHAP top 5:            ", ", ".join(mean_abs.sort_values(ascending=False).head(5).index))
    print("  Agreement between the two rankings (1 = identical): {:.2f}".format(spearman))

    # 1. Bar plot: average size of each feature's effect
    plt.figure()
    shap.plots.bar(sv, max_display=12, show=False)
    save("shap_{}_1_bar.png".format(task))

    # 2. Summary (beeswarm) plot: direction and size of effects
    plt.figure()
    shap.plots.beeswarm(sv, max_display=12, show=False)
    save("shap_{}_2_summary.png".format(task))

    # 3. Dependence plot for the most important feature
    top = mean_abs.idxmax()
    plt.figure()
    shap.plots.scatter(sv[:, top], show=False)
    save("shap_{}_3_dependence_{}.png".format(task, top))

    # 4. Force plot for one example row (the last validation bar)
    try:
        shap.plots.force(sv[-1], matplotlib=True, show=False)
        save("shap_{}_4_force.png".format(task))
    except Exception as e:
        print("  (force plot skipped: {})".format(type(e).__name__))
        plt.close("all")

print("\nSaved SHAP charts in:", PLOTS)

print("\n" + "=" * 70)
print("VALIDATION RESULTS SO FAR (saved in results_all.csv)")
print("=" * 70)
show_table("validation")
