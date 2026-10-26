"""
Step 2 (new): Get the professor's Bitstamp data and turn it into hourly bars
Run from inside the "Bitcoin MLBD" folder:   python get_bitstamp_data.py
Saves: BitstampData_sample.csv (raw, minute data) and bitstamp_hourly.csv (hourly bars)
"""

import os
import urllib.request
import pandas as pd

URL = ("https://raw.githubusercontent.com/tiwvis/MLBD-Projects/main/"
       "data_sets/BitstampData_sample.csv")

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(HERE, "BitstampData_sample.csv")
OUT_PATH = os.path.join(HERE, "bitstamp_hourly.csv")

# 1. Download the raw file (skipped if you already have it in this folder)
if not os.path.exists(RAW_PATH):
    print("Downloading the dataset (about 20 MB)...")
    try:
        urllib.request.urlretrieve(URL, RAW_PATH)
    except Exception as e:
        raise SystemExit(
            "Download failed: {}\nOpen the link below in your browser, save the file as "
            "BitstampData_sample.csv in this folder, then run this script again.\n{}".format(e, URL))

raw = pd.read_csv(RAW_PATH)
raw["Date"] = pd.to_datetime(raw["Timestamp"], unit="s")
raw = raw.set_index("Date")

print("Raw file")
print("  Rows (one per minute):", len(raw))
print("  Date range:", raw.index.min(), "to", raw.index.max())
print("  Minutes with no trades (empty rows): {:.1f}%".format(raw["Close"].isna().mean() * 100))

# 2. Keep only minutes where trades actually happened
trades = raw.dropna(subset=["Close"]).rename(
    columns={"Volume_(BTC)": "Volume_BTC", "Volume_(Currency)": "Volume_USD"})
print("  Minutes with trades:", len(trades))

# 3. Combine minutes into hourly bars
hourly = trades.resample("1h").agg({
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume_BTC": "sum",
    "Volume_USD": "sum",
})
hourly["Active_minutes"] = trades["Close"].resample("1h").count()  # how many minutes had a trade

total_hours = len(hourly)
hourly = hourly[hourly["Active_minutes"] > 0]   # drop hours where nothing traded
hourly.index.name = "Date"
hourly.to_csv(OUT_PATH)

# 4. Quick check
print("\nHourly bars")
print("  Hours in the period:", total_hours)
print("  Hours with trades (kept):", len(hourly))
print("  Hours with no trades (dropped): {}".format(total_hours - len(hourly)))
print("  Shape (rows, columns):", hourly.shape)
print("  Price range: {:.2f} to {:.2f} USD".format(hourly["Low"].min(), hourly["High"].max()))
print()
print(hourly.head())
print()
print("Missing values per column:")
print(hourly.isna().sum())
print("\nSaved to:", OUT_PATH)
