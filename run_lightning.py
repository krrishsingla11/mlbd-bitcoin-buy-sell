"""
Step 6e: PyTorch Lightning (regression + Buy/Sell now + Buy/Sell 12 bars ahead), on VALIDATION
(Used instead of Lightning Flash, which is no longer maintained and cannot be installed
 on current Python versions. Flash was built on top of PyTorch Lightning.)

Run from inside the "Bitcoin MLBD" folder:   python run_lightning.py
Needs bs_train.csv, bs_val.csv, bs_test.csv (Step 5) and common.py in the same folder.
Install first:  pip install lightning
The test set is not touched here.
"""

import time
import logging
import tempfile
import warnings
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from common import (load_data, scale_with_train, reg_metrics, clf_metrics,
                    log_result, log_baselines, show_table, SEED)

warnings.filterwarnings("ignore")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)   # keep the screen clean

train, val, test, FEATURES = load_data()
log_baselines(train, val)

X_train, X_val = scale_with_train(train[FEATURES], val[FEATURES])
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_val = torch.tensor(X_val.values, dtype=torch.float32)


class TabularNet(pl.LightningModule):
    """Same small network as the Keras and fastai versions: 64 -> 32 -> 1, with dropout."""

    def __init__(self, n_inputs, task):
        super().__init__()
        self.task = task
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1))
        self.loss_fn = nn.MSELoss() if task == "regression" else nn.BCEWithLogitsLoss()
        self.val_losses = []

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def training_step(self, batch, _):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=False)

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            self.val_losses.append(float(self.trainer.callback_metrics["val_loss"]))

    def predict_step(self, batch, _):
        out = self(batch[0])
        return out if self.task == "regression" else torch.sigmoid(out)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)


def run(task, y_train, y_val):
    """Train one model with early stopping and return validation predictions."""
    pl.seed_everything(SEED, workers=True)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True,
                          generator=torch.Generator().manual_seed(SEED))
    val_dl = DataLoader(TensorDataset(X_val, y_val), batch_size=256)

    model = TabularNet(X_train.shape[1], task)
    n_w = sum(p.numel() for p in model.parameters())

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = ModelCheckpoint(dirpath=tmp, monitor="val_loss", mode="min", save_top_k=1)
        trainer = pl.Trainer(max_epochs=200, accelerator="cpu", devices=1,
                             callbacks=[EarlyStopping(monitor="val_loss", patience=15), ckpt],
                             logger=False, enable_progress_bar=False,
                             enable_model_summary=False)
        t0 = time.time()
        trainer.fit(model, train_dl, val_dl)
        secs = time.time() - t0
        preds = trainer.predict(model, DataLoader(TensorDataset(X_val, y_val), batch_size=256),
                                ckpt_path="best")          # use the best epoch, not the last
    best_epoch = int(np.argmin(model.val_losses)) + 1
    return torch.cat(preds).numpy(), secs, best_epoch, n_w


# ---------- 1. Regression: next-bar return ----------
y_mean, y_std = train["target_ret_clip"].mean(), train["target_ret_clip"].std()
y_tr = ((train["target_ret_clip"] - y_mean) / y_std).values
y_va = ((val["target_ret_clip"] - y_mean) / y_std).values
preds, secs, best, n_w = run("regression", y_tr, y_va)
pred = preds * y_std + y_mean
log_result("Lightning", "regression", "validation", reg_metrics(val["target_ret_clip"], pred), secs,
           "{} epochs, {} weights".format(best, n_w))
print("Regression done: best at epoch {}, {:.1f}s".format(best, secs))

# ---------- 2 and 3. Classification ----------
for task, col, label in [("class_now", "target_class", "Buy/Sell now"),
                         ("class_12", "target_class_12", "Buy/Sell 12 bars ahead")]:
    prob, secs, best, n_w = run(task, train[col].values, val[col].values)
    log_result("Lightning", task, "validation", clf_metrics(val[col], prob), secs,
               "{} epochs, {} weights".format(best, n_w))
    print("Classification ({}) done: best at epoch {}, {:.1f}s".format(label, best, secs))

print("\n" + "=" * 70)
print("VALIDATION RESULTS SO FAR (saved in results_all.csv)")
print("=" * 70)
show_table("validation")
