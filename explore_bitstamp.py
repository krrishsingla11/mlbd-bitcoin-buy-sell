"""
Step 3 (new): Explore the hourly Bitstamp data (EDA)
Run from inside the "Bitcoin MLBD" folder:   python explore_bitstamp.py
Needs bitstamp_hourly.csv from Step 2. Saves charts into a "plots" folder.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, "plots")
os.makedirs(PLOTS, exist_ok=True)

SHORT, LONG = 10, 60   # short-term and long-term window (in bars), used for the Buy/Sell signal

df = pd.read_csv(os.path.join(HERE, "bitstamp_hourly.csv"), index_col="Date", parse_dates=True)

# ---------- 1. Basic checks ----------
print("=" * 55)
print("1. BASIC CHECKS")
print("=" * 55)
print("Rows:", len(df))
print("Date range:", df.index.min(), "to", df.index.max())
print("Duplicate timestamps:", int(df.index.duplicated().sum()))
print("Missing values:", int(df.isna().sum().sum()))
print("Rows where High < Low (bad data):", int((df["High"] < df["Low"]).sum()))
print()
print(df.describe().round(2).T[["mean", "std", "min", "50%", "max"]])

# ---------- 2. Gaps in time (hours with no trades were dropped) ----------
gaps = df.index.to_series().diff().dt.total_seconds().div(3600).dropna()
print()
print("=" * 55)
print("2. GAPS BETWEEN BARS")
print("=" * 55)
print("Typical gap: {:.0f} hour(s)".format(gaps.median()))
print("Bars that follow a gap of 2+ hours: {} ({:.1f}%)".format((gaps >= 2).sum(), (gaps >= 2).mean() * 100))
print("Longest gap: {:.0f} hours ({:.1f} days)".format(gaps.max(), gaps.max() / 24))
bars_per_month = df["Close"].resample("MS").count()
print("\nBars per month (activity over time):")
print(bars_per_month.to_string())

# ---------- 3. Returns ----------
df["ret"] = df["Close"].pct_change()
print()
print("=" * 55)
print("3. RETURNS PER BAR")
print("=" * 55)
print("Average return: {:.3f}%".format(df["ret"].mean() * 100))
print("Typical swing (std): {:.2f}%".format(df["ret"].std() * 100))
print("Biggest gain: {:.1f}% on {}".format(df["ret"].max() * 100, df["ret"].idxmax()))
print("Biggest drop: {:.1f}% on {}".format(df["ret"].min() * 100, df["ret"].idxmin()))
print("Bars with a move bigger than 5%:", int((df["ret"].abs() > 0.05).sum()))
print("Bars with exactly zero move: {:.1f}%".format((df["ret"] == 0).mean() * 100))

# ---------- 4. Buy/Sell signal (professor's definition) ----------
short_ma = df["Close"].rolling(SHORT).mean()
long_ma = df["Close"].rolling(LONG).mean()
valid = long_ma.notna()
signal = (short_ma > long_ma).astype(int)[valid]
next_signal = signal.shift(-1).dropna()
same_as_now = (signal.iloc[:-1].values == next_signal.values).mean()

print()
print("=" * 55)
print("4. BUY/SELL SIGNAL (short {} bars vs long {} bars)".format(SHORT, LONG))
print("=" * 55)
print("Buy (1):  {} ({:.1f}%)".format((signal == 1).sum(), (signal == 1).mean() * 100))
print("Sell (0): {} ({:.1f}%)".format((signal == 0).sum(), (signal == 0).mean() * 100))
print("Signal flips between one bar and the next: {} times".format(int((signal.diff().abs() == 1).sum())))
print("Next bar's signal is the SAME as the current one: {:.1f}% of the time".format(same_as_now * 100))
print("-> A model that just repeats the current signal would score {:.1f}% accuracy.".format(same_as_now * 100))
print("   That is the real baseline to beat, not 50%.")

# ---------- 5. Charts ----------
# Price
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(df.index, df["Close"], linewidth=0.8, label="Close")
ax.plot(short_ma.index, short_ma, linewidth=0.8, label="{}-bar average".format(SHORT))
ax.plot(long_ma.index, long_ma, linewidth=0.8, label="{}-bar average".format(LONG))
ax.set_title("Bitcoin price on Bitstamp (hourly bars)")
ax.set_ylabel("USD")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "1_price.png"), dpi=150)
plt.close(fig)

# Returns over time
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(df.index, df["ret"] * 100, linewidth=0.5)
ax.set_title("Return per bar (%)")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "2_returns.png"), dpi=150)
plt.close(fig)

# Return distribution (clipped so a few extreme bars don't squash the chart)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df["ret"].dropna().clip(-0.1, 0.1) * 100, bins=80)
ax.set_title("How big are the moves? (%, clipped at +/-10%)")
ax.set_xlabel("Return per bar (%)")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "3_return_hist.png"), dpi=150)
plt.close(fig)

# Volume
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(df.index, df["Volume_BTC"], linewidth=0.5)
ax.set_yscale("log")
ax.set_title("Trading volume per bar (BTC, log scale)")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "4_volume.png"), dpi=150)
plt.close(fig)

# Data density: bars per month
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.bar(bars_per_month.index.strftime("%b %y"), bars_per_month.values)
ax.set_title("Bars with trades per month (market activity grows)")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "5_bars_per_month.png"), dpi=150)
plt.close(fig)

# Buy vs Sell
fig, ax = plt.subplots(figsize=(4, 4))
signal.value_counts().sort_index().rename({0: "Sell", 1: "Buy"}).plot(kind="bar", ax=ax)
ax.set_title("Buy vs Sell signal")
ax.set_xlabel("")
plt.setp(ax.get_xticklabels(), rotation=0)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "6_class_balance.png"), dpi=150)
plt.close(fig)

# Correlation heatmap
cols = ["Open", "High", "Low", "Close", "Volume_BTC", "Volume_USD", "Active_minutes", "ret"]
corr = df[cols].corr()
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
ax.set_xticks(range(len(cols)))
ax.set_yticks(range(len(cols)))
ax.set_xticklabels(cols, rotation=45, ha="right")
ax.set_yticklabels(cols)
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, "{:.2f}".format(corr.iloc[i, j]), ha="center", va="center", fontsize=7)
fig.colorbar(im)
ax.set_title("Correlation heatmap")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "7_correlation.png"), dpi=150)
plt.close(fig)

print()
print("Saved 7 charts in:", PLOTS)
