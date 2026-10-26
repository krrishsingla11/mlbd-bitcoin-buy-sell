"""
Step 8: FINAL test-set evaluation of all five frameworks + the blend
Run from inside the "Bitcoin MLBD" folder:   python run_final_test.py
Needs bs_train.csv, bs_val.csv, bs_test.csv (Step 5) and the updated common.py.

Every model is trained exactly as before (train set, early stopping on validation) and
then scored on the TEST set. Run this ONCE and do not change anything afterwards because
of what the test scores say, otherwise the test set stops being an honest check.
Takes about 5 to 10 minutes. Saves test_results_final.csv and results_all.csv.
"""

import io
import time
import logging
import tempfile
import warnings
from contextlib import redirect_stdout
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import keras
from keras import layers, regularizers
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from xgboost import XGBRegressor, XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from fastai.tabular.all import (TabularPandas, Normalize, CategoryBlock, RegressionBlock,
                                tabular_learner, tabular_config, EarlyStoppingCallback,
                                SaveModelCallback, set_seed)
from common import (HERE, load_data, scale_with_train, reg_metrics, clf_metrics,
                    log_result, log_baselines, show_table, RESULTS_PATH, SEED)
import os

warnings.filterwarnings("ignore")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

train, val, test, FEATURES = load_data()
Xs_train, Xs_val, Xs_test = scale_with_train(train[FEATURES], val[FEATURES], test[FEATURES])
Xs_train, Xs_val, Xs_test = [x.values.astype("float32") for x in (Xs_train, Xs_val, Xs_test)]

TASKS = [("regression", "target_ret_clip", "reg"),
         ("class_now", "target_class", "clf"),
         ("class_12", "target_class_12", "clf")]

XGB = dict(n_estimators=1000, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.8,
           min_child_weight=5, early_stopping_rounds=50, random_state=SEED)


# ------------------------------------------------------------------ XGBoost
def fit_xgboost(kind, ytr, yva):
    if kind == "reg":
        m = XGBRegressor(eval_metric="rmse", **XGB)
        f = lambda X: m.predict(X)
    else:
        m = XGBClassifier(eval_metric="logloss", **XGB)
        f = lambda X: m.predict_proba(X)[:, 1]
    m.fit(train[FEATURES], ytr, eval_set=[(val[FEATURES], yva)], verbose=False)
    return f(val[FEATURES]), f(test[FEATURES])


# ------------------------------------------------------------------ Keras
def fit_keras(kind, ytr, yva):
    keras.utils.set_random_seed(SEED)
    net = keras.Sequential([
        keras.Input(shape=(len(FEATURES),)),
        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        layers.Dense(1, activation="linear" if kind == "reg" else "sigmoid"),
    ])
    net.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse" if kind == "reg" else "binary_crossentropy")
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    net.fit(Xs_train, ytr.astype("float32"), validation_data=(Xs_val, yva.astype("float32")),
            epochs=200, batch_size=64, callbacks=[stop], verbose=0)
    return net.predict(Xs_val, verbose=0).ravel(), net.predict(Xs_test, verbose=0).ravel()


# ------------------------------------------------------------------ fastai
def fit_fastai(kind, ytr, yva):
    df = pd.concat([train[FEATURES], val[FEATURES]]).reset_index(drop=True)
    y = np.concatenate([ytr, yva])
    df["y"] = y.astype("float32") if kind == "reg" else y.astype(int)
    n = len(train)
    to = TabularPandas(df, procs=[Normalize], cat_names=[], cont_names=FEATURES, y_names="y",
                       y_block=RegressionBlock() if kind == "reg" else CategoryBlock(),
                       splits=(list(range(n)), list(range(n, len(df)))))
    dls = to.dataloaders(bs=64, num_workers=0)
    set_seed(SEED, reproducible=True)
    learn = tabular_learner(dls, layers=[64, 32], config=tabular_config(ps=[0.3, 0.3]))
    cbs = [EarlyStoppingCallback(monitor="valid_loss", patience=15),
           SaveModelCallback(monitor="valid_loss", fname="final_" + kind + str(len(ytr)))]
    with learn.no_bar(), learn.no_logging(), redirect_stdout(io.StringIO()):
        learn.fit(200, lr=1e-3, wd=1e-4, cbs=cbs)

    def predict(X):
        dl = learn.dls.test_dl(X)
        with learn.no_bar(), redirect_stdout(io.StringIO()):
            p, _ = learn.get_preds(dl=dl)
        p = p.numpy()
        return p.ravel() if kind == "reg" else p[:, 1]

    return predict(val[FEATURES]), predict(test[FEATURES])


# ------------------------------------------------------------------ TabNet
def fit_tabnet(kind, ytr, yva):
    torch.manual_seed(SEED)
    params = dict(n_d=8, n_a=8, n_steps=3, gamma=1.3, n_independent=1, n_shared=1,
                  optimizer_params=dict(lr=1e-2), seed=SEED, verbose=0)
    fit_args = dict(max_epochs=200, patience=15, batch_size=256, virtual_batch_size=64)
    with redirect_stdout(io.StringIO()):
        if kind == "reg":
            m = TabNetRegressor(**params)
            m.fit(Xs_train, ytr.reshape(-1, 1).astype("float32"),
                  eval_set=[(Xs_val, yva.reshape(-1, 1).astype("float32"))], eval_metric=["rmse"], **fit_args)
            f = lambda X: m.predict(X).ravel()
        else:
            m = TabNetClassifier(**params)
            m.fit(Xs_train, ytr, eval_set=[(Xs_val, yva)], eval_metric=["logloss"], **fit_args)
            f = lambda X: m.predict_proba(X)[:, 1]
    return f(Xs_val), f(Xs_test)


# ------------------------------------------------------------------ PyTorch Lightning
class TabularNet(pl.LightningModule):
    def __init__(self, n_inputs, kind):
        super().__init__()
        self.kind = kind
        self.net = nn.Sequential(nn.Linear(n_inputs, 64), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))
        self.loss_fn = nn.MSELoss() if kind == "reg" else nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def training_step(self, batch, _):
        x, y = batch
        return self.loss_fn(self(x), y)

    def validation_step(self, batch, _):
        x, y = batch
        self.log("val_loss", self.loss_fn(self(x), y), on_epoch=True)

    def predict_step(self, batch, _):
        out = self(batch[0])
        return out if self.kind == "reg" else torch.sigmoid(out)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)


def fit_lightning(kind, ytr, yva):
    pl.seed_everything(SEED, workers=True)
    Xt, Xv, Xte = [torch.tensor(x) for x in (Xs_train, Xs_val, Xs_test)]
    yt, yv = torch.tensor(ytr, dtype=torch.float32), torch.tensor(yva, dtype=torch.float32)
    train_dl = DataLoader(TensorDataset(Xt, yt), batch_size=64, shuffle=True,
                          generator=torch.Generator().manual_seed(SEED))
    val_dl = DataLoader(TensorDataset(Xv, yv), batch_size=256)
    model = TabularNet(Xt.shape[1], kind)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = ModelCheckpoint(dirpath=tmp, monitor="val_loss", mode="min", save_top_k=1)
        trainer = pl.Trainer(max_epochs=200, accelerator="cpu", devices=1, logger=False,
                             callbacks=[EarlyStopping(monitor="val_loss", patience=15), ckpt],
                             enable_progress_bar=False, enable_model_summary=False)
        trainer.fit(model, train_dl, val_dl)
        pv = trainer.predict(model, DataLoader(TensorDataset(Xv), batch_size=256), ckpt_path="best")
        pt = trainer.predict(model, DataLoader(TensorDataset(Xte), batch_size=256), ckpt_path="best")
    return torch.cat(pv).numpy(), torch.cat(pt).numpy()


FRAMEWORKS = [("XGBoost", fit_xgboost), ("Keras", fit_keras), ("fastai", fit_fastai),
              ("TabNet", fit_tabnet), ("Lightning", fit_lightning)]

# ------------------------------------------------------------------ run everything
log_baselines(train, val, "validation")
log_baselines(train, test, "test")

for task, col, kind in TASKS:
    ytr_raw, yva_raw = train[col].values, val[col].values
    if kind == "reg":
        mu, sd = ytr_raw.mean(), ytr_raw.std()
        ytr_s, yva_s = (ytr_raw - mu) / sd, (yva_raw - mu) / sd
    preds = {}
    for name, fn in FRAMEWORKS:
        t0 = time.time()
        if name == "XGBoost" or kind == "clf":
            pv, pt = fn(kind, ytr_raw, yva_raw)
        else:                                           # neural networks learn a scaled target
            pv, pt = fn(kind, ytr_s, yva_s)
            pv, pt = pv * sd + mu, pt * sd + mu
        preds[name] = (pv, pt, time.time() - t0)
        print("  {:<10} {:<10} done in {:.1f}s".format(task, name, time.time() - t0))

    xv, xt, xs = preds["XGBoost"]
    kv, kt, ks = preds["Keras"]
    preds["Blend (XGB+Keras)"] = ((xv + kv) / 2, (xt + kt) / 2, xs + ks)

    for name, (pv, pt, secs) in preds.items():
        if kind == "reg":
            log_result(name, task, "validation", reg_metrics(val[col], pv), secs, "final run")
            log_result(name, task, "test", reg_metrics(test[col], pt), secs, "final run")
            log_result(name, "regression_raw", "test", reg_metrics(test["target_ret"], pt), secs, "final run")
        else:
            log_result(name, task, "validation", clf_metrics(val[col], pv), secs, "final run")
            log_result(name, task, "test", clf_metrics(test[col], pt), secs, "final run")

# ------------------------------------------------------------------ report
print("\n" + "=" * 72)
print("FINAL TEST RESULTS (the locked test set, Nov 2012 to Dec 2012)")
print("=" * 72)
show_table("test")

res = pd.read_csv(RESULTS_PATH)
res[res["Split"] == "test"].to_csv(os.path.join(HERE, "test_results_final.csv"), index=False)

print("\n" + "=" * 72)
print("VALIDATION -> TEST  (main score per task; a big drop means the model did not hold up)")
print("=" * 72)
for task, metric in [("regression", "R2"), ("class_now", "Accuracy"), ("class_12", "Accuracy")]:
    t = res[res["Task"] == task].pivot(index="Framework", columns="Split", values=metric)
    t = t[["validation", "test"]].round(4)
    t["change"] = (t["test"] - t["validation"]).round(4)
    print("\n[{}]  {}".format(task, metric))
    print(t.to_string())

print("\nSaved: test_results_final.csv and results_all.csv")
