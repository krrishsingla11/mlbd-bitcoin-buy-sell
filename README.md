# Predicting Buy or Sell in the Bitcoin Market (MLBD Project Id 3)

Hourly Bitstamp Bitcoin data (Dec 2011 to Dec 2012). Buy/Sell classification and next-hour return regression
with XGBoost, Keras, PyTorch + fastai, PyTorch + TabNet and PyTorch Lightning
(Lightning is used in place of the unmaintained Lightning Flash).

## Team
- Krrish Singla (PES1UG24AM141)
- Shesanth K (PES1UG24AM261)
- Shamika Mahesh (PES1UG24AM259)

## Setup
    pip install -r requirements.txt

## Run in this order
| Step | Script | What it does |
|---|---|---|
| 1 | get_bitstamp_data.py | Downloads the Bitstamp sample and builds hourly bars (bitstamp_hourly.csv) |
| 2 | explore_bitstamp.py | EDA checks and charts (plots/) |
| 3 | build_features_bitstamp.py | 26 features and the targets (bitstamp_features.csv) |
| 4 | split_bitstamp.py | Time-ordered train / validation / test split (bs_*.csv) |
| 5 | run_xgboost.py, run_keras.py, run_fastai.py, run_tabnet.py, run_lightning.py | One framework each, scored on validation |
| 6 | run_blend_shap.py | XGBoost + Keras blend and SHAP explanations |
| 7 | run_final_test.py | Final scores on the held-out test set (test_results_final.csv) |

common.py holds the shared data loading, metrics and results logging.

## Targets
- Buy/Sell now: 1 if the 10-bar average price is above the 60-bar average
- Buy/Sell 12 bars ahead: the same signal, 12 bars later (extra experiment)
- Expected return: next bar's return, capped at +/-5%
