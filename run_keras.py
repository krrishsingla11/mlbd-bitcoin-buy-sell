"""
Step 6b: Keras neural network (regression + Buy/Sell now + Buy/Sell 12 bars ahead), on VALIDATION
Run from inside the "Bitcoin MLBD" folder:   python run_keras.py
Needs bs_train.csv, bs_val.csv, bs_test.csv (Step 5) and common.py in the same folder.
Install first:  pip install tensorflow
The test set is not touched here.
"""

import time
import warnings
import numpy as np
import keras
from keras import layers, regularizers
from common import (load_data, scale_with_train, reg_metrics, clf_metrics,
                    log_result, log_baselines, show_table, SEED)

warnings.filterwarnings("ignore")

train, val, test, FEATURES = load_data()
log_baselines(train, val)

# Neural networks need inputs on a similar scale: standardise using TRAIN statistics only
X_train, X_val = scale_with_train(train[FEATURES], val[FEATURES])
X_train, X_val = X_train.values.astype("float32"), X_val.values.astype("float32")


def build_model(n_inputs, task):
    """A small network: 2 hidden layers, dropout and L2 penalty to fight overfitting."""
    model = keras.Sequential([
        keras.Input(shape=(n_inputs,)),
        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        layers.Dense(1, activation="linear" if task == "regression" else "sigmoid"),
    ])
    if task == "regression":
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    else:
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return model


def fit(model, y_tr, y_va):
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    model.fit(X_train, y_tr, validation_data=(X_val, y_va),
              epochs=200, batch_size=64, callbacks=[stop], verbose=0)
    return stop.best_epoch + 1      # the epoch whose weights we keep


# ---------- 1. Regression: next-bar return ----------
keras.utils.set_random_seed(SEED)
y_mean, y_std = train["target_ret_clip"].mean(), train["target_ret_clip"].std()
y_tr = ((train["target_ret_clip"] - y_mean) / y_std).values.astype("float32")   # scale the target too
y_va = ((val["target_ret_clip"] - y_mean) / y_std).values.astype("float32")

t0 = time.time()
model = build_model(len(FEATURES), "regression")
epochs = fit(model, y_tr, y_va)
secs = time.time() - t0
pred = model.predict(X_val, verbose=0).ravel() * y_std + y_mean                  # back to real returns
log_result("Keras", "regression", "validation", reg_metrics(val["target_ret_clip"], pred), secs,
           "{} epochs, {} weights".format(epochs, model.count_params()))
print("Regression done: best at epoch {}, {:.1f}s".format(epochs, secs))

# ---------- 2 and 3. Classification ----------
for task, col, label in [("class_now", "target_class", "Buy/Sell now"),
                         ("class_12", "target_class_12", "Buy/Sell 12 bars ahead")]:
    keras.utils.set_random_seed(SEED)
    t0 = time.time()
    model = build_model(len(FEATURES), task)
    epochs = fit(model, train[col].values.astype("float32"), val[col].values.astype("float32"))
    secs = time.time() - t0
    prob = model.predict(X_val, verbose=0).ravel()
    log_result("Keras", task, "validation", clf_metrics(val[col], prob), secs,
               "{} epochs, {} weights".format(epochs, model.count_params()))
    print("Classification ({}) done: best at epoch {}, {:.1f}s".format(label, epochs, secs))

print("\n" + "=" * 70)
print("VALIDATION RESULTS SO FAR (saved in results_all.csv)")
print("=" * 70)
show_table("validation")
