"""
Step 4 (new): Build features and targets from the hourly Bitstamp bars
Run from inside the "Bitcoin MLBD" folder:   python build_features_bitstamp.py
Needs bitstamp_hourly.csv from Step 2. Saves bitstamp_features.csv.

Targets saved in the file (these are NEVER used as inputs):
  target_class     Buy(1)/Sell(0) = is the 10-bar average above the 60-bar average NOW   (main task)
  target_class_12  the same Buy/Sell signal, but 12 bars in the future                    (extra experiment)
  target_ret       the NEXT bar's return = the "expected return" (regression, raw)
  target_ret_clip  the same, capped at +/-5% so a few thin-market spikes don't dominate
"""

import os
import numpy as np
import pandas as pd

SHORT, LONG = 10, 60
HORIZON = 12          # how far ahead the extra experiment looks (in bars)
CLIP = 0.05           # cap for the regression target

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "bitstamp_hourly.csv"), index_col="Date", parse_dates=True)

close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume_BTC"]
feat = pd.DataFrame(index=df.index)

# ---------- Features (each uses only the current and earlier bars) ----------
# Returns: recent price moves
ret = close.pct_change()
feat["ret_1"] = ret
feat["ret_lag1"] = ret.shift(1)
feat["ret_lag2"] = ret.shift(2)
feat["ret_3"] = close.pct_change(3)
feat["ret_6"] = close.pct_change(6)
feat["ret_12"] = close.pct_change(12)

# Trend: price vs exponential averages (windows 5, 20, 30 on purpose, NOT 10 or 60)
for n in (5, 20, 30):
    feat["close_vs_ema{}".format(n)] = close / close.ewm(span=n, adjust=False).mean() - 1

# Momentum
feat["roc_20"] = close.pct_change(20)
delta = close.diff()
gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
feat["rsi_14"] = 100 - 100 / (1 + gain / loss)

low14, high14 = low.rolling(14).min(), high.rolling(14).max()
feat["stoch_k_14"] = (close - low14) / (high14 - low14)
feat["stoch_d_3"] = feat["stoch_k_14"].rolling(3).mean()

ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()
macd = ema12 - ema26
macd_signal = macd.ewm(span=9, adjust=False).mean()
feat["macd"] = macd / close
feat["macd_signal"] = macd_signal / close
feat["macd_hist"] = (macd - macd_signal) / close

# Volatility and candle shape
feat["vol_10"] = ret.rolling(10).std()
feat["vol_20"] = ret.rolling(20).std()
feat["range_pct"] = (high - low) / close
feat["body_pct"] = (close - df["Open"]) / df["Open"]

# Volume and market activity
feat["volume_change"] = vol.pct_change().replace([np.inf, -np.inf], np.nan)
feat["volume_vs_ma20"] = vol / vol.rolling(20).mean() - 1
feat["active_minutes"] = df["Active_minutes"]
feat["gap_hours"] = df.index.to_series().diff().dt.total_seconds() / 3600

# Time of day (as a circle, so 23:00 and 00:00 are close together)
hour = df.index.hour
feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)

# ---------- Targets ----------
short_ma = close.rolling(SHORT).mean()
long_ma = close.rolling(LONG).mean()
signal = (short_ma > long_ma).astype(float).where(long_ma.notna())

feat["target_class"] = signal
feat["target_class_12"] = signal.shift(-HORIZON)
next_ret = close.shift(-1) / close - 1
feat["target_ret"] = next_ret
feat["target_ret_clip"] = next_ret.clip(-CLIP, CLIP)

# ---------- Clean ----------
before = len(feat)
feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
for c in ["target_class", "target_class_12"]:
    feat[c] = feat[c].astype(int)
feat.to_csv(os.path.join(HERE, "bitstamp_features.csv"))

# ---------- Report ----------
targets = ["target_class", "target_class_12", "target_ret", "target_ret_clip"]
inputs = [c for c in feat.columns if c not in targets]

print("Rows before cleaning:", before)
print("Rows after cleaning: ", len(feat), "(first {} bars need history, last {} have no future)".format(LONG, HORIZON))
print("Date range:", feat.index.min(), "to", feat.index.max())
print("Number of input features:", len(inputs))
print("Features:", inputs)

print("\n--- Classification (main): Buy/Sell now ---")
print("Buy (1): {:.1f}%   Sell (0): {:.1f}%   -> 'always Buy' accuracy: {:.1f}%".format(
    feat["target_class"].mean() * 100, (1 - feat["target_class"].mean()) * 100, feat["target_class"].mean() * 100))

print("\n--- Classification (extra): Buy/Sell {} bars ahead ---".format(HORIZON))
print("Buy (1): {:.1f}%   Sell (0): {:.1f}%".format(
    feat["target_class_12"].mean() * 100, (1 - feat["target_class_12"].mean()) * 100))
print("'Repeat the current signal' accuracy: {:.1f}%  <- baseline to beat".format(
    (feat["target_class"] == feat["target_class_12"]).mean() * 100))

print("\n--- Regression: next-bar return ---")
print("Average: {:.3f}%   Std: {:.2f}%".format(feat["target_ret"].mean() * 100, feat["target_ret"].std() * 100))
print("Bars capped at +/-{:.0f}%: {} ({:.1f}%)".format(
    CLIP * 100, int((feat["target_ret"].abs() > CLIP).sum()), (feat["target_ret"].abs() > CLIP).mean() * 100))
print("Baseline: always predict the average return -> R2 = 0")

print("\n--- Link between each feature and the targets (0 = none, 1 = perfect) ---")
corr = feat.corr()[["target_class", "target_ret_clip"]].drop(index=targets).round(3)
corr["abs_class"] = corr["target_class"].abs()
print(corr.sort_values("abs_class", ascending=False).drop(columns="abs_class").to_string())

print("\nSaved to:", os.path.join(HERE, "bitstamp_features.csv"))
