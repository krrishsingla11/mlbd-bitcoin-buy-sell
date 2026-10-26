"""
Step 5 (new): Split the Bitstamp features by time
Run from inside the "Bitcoin MLBD" folder:   python split_bitstamp.py
Needs bitstamp_features.csv (Step 4) and bitstamp_hourly.csv (Step 2).
Saves bs_train.csv, bs_val.csv, bs_test.csv.

Oldest 70% -> train, next 15% -> validation, newest 15% -> test (never shuffled).
A 12-bar gap is left at the end of train and validation, because the "12 bars ahead"
target looks 12 bars into the future and would otherwise peek into the next set.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

GAP = 12

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "bitstamp_features.csv"), index_col="Date", parse_dates=True).sort_index()

n = len(df)
a, b = int(n * 0.70), int(n * 0.85)

train = df.iloc[: a - GAP]
val = df.iloc[a: b - GAP]
test = df.iloc[b:]

train.to_csv(os.path.join(HERE, "bs_train.csv"))
val.to_csv(os.path.join(HERE, "bs_val.csv"))
test.to_csv(os.path.join(HERE, "bs_test.csv"))

# ---------- Report ----------
print("Rows: train {}, validation {}, test {}  (total {}, {} dropped as gaps)\n".format(
    len(train), len(val), len(test), n, 2 * GAP))

print("{:<12}{:<26}{:>8}{:>16}{:>18}{:>16}".format(
    "Set", "Dates", "Rows", "Buy % (now)", "Repeat-signal acc", "Return std %"))
for name, part in [("Train", train), ("Validation", val), ("Test", test)]:
    repeat = (part["target_class"] == part["target_class_12"]).mean() * 100
    print("{:<12}{:<26}{:>8}{:>15.1f}%{:>17.1f}%{:>15.2f}%".format(
        name,
        "{} to {}".format(part.index.min().date(), part.index.max().date()),
        len(part),
        part["target_class"].mean() * 100,
        repeat,
        part["target_ret_clip"].std() * 100))

print("\n'Buy % (now)' is also the accuracy of always predicting Buy.")
print("'Repeat-signal acc' is the baseline for the 12-bars-ahead experiment.")

assert train.index.max() < val.index.min() < val.index.max() < test.index.min()
print("\nCheck passed: train < validation < test, no overlap.")

# ---------- Picture ----------
raw = pd.read_csv(os.path.join(HERE, "bitstamp_hourly.csv"), index_col="Date", parse_dates=True)
fig, ax = plt.subplots(figsize=(11, 4))
for part, label, color in [(train, "Train", "tab:blue"), (val, "Validation", "tab:orange"), (test, "Test", "tab:green")]:
    seg = raw.loc[part.index.min(): part.index.max(), "Close"]
    ax.plot(seg.index, seg, label=label, color=color, linewidth=0.8)
ax.set_title("Time-based split of the Bitstamp data")
ax.set_ylabel("Close (USD)")
ax.legend()
fig.tight_layout()
os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)
fig.savefig(os.path.join(HERE, "plots", "8_split.png"), dpi=150)
print("Saved chart: plots/8_split.png")
