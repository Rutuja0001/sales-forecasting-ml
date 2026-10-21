"""
Project 8: Sales Forecasting (time series)
Gradient Boosting on calendar + lag + rolling features, evaluated on the last
90 days against a seasonal-naive baseline, then a 30-day recursive forecast.

Required CSV columns (for your own data): date, sales, promo (0/1)

Run: python sales_forecasting.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

OUT = Path(__file__).resolve().parent
TEST_DAYS, HORIZON = 90, 30
FEATS = ["dow", "month", "doy", "promo", "lag_1", "lag_7", "lag_14", "lag_28", "roll7", "roll28"]


def generate_data(seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=1095, freq="D")
    t = np.arange(len(dates))
    weekly = np.array([-10, -8, -5, 0, 10, 35, 30])[dates.dayofweek]
    promo = (rng.random(len(dates)) < 0.05).astype(int)
    sales = (200 + 0.15 * t + 25 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
             + weekly + 40 * promo + rng.normal(0, 8, len(dates)))
    return pd.DataFrame({"date": dates, "sales": sales.round(1), "promo": promo})


def build_features(df):
    d = df.copy()
    d["dow"], d["month"], d["doy"] = d.date.dt.dayofweek, d.date.dt.month, d.date.dt.dayofyear
    for lag in (1, 7, 14, 28):
        d[f"lag_{lag}"] = d["sales"].shift(lag)
    d["roll7"] = d["sales"].shift(1).rolling(7).mean()
    d["roll28"] = d["sales"].shift(1).rolling(28).mean()
    return d


def forecast(model, hist, days=HORIZON):
    """Recursive multi-step forecast: each prediction feeds the next day's lags."""
    hist = hist[["date", "sales", "promo"]].copy()
    for _ in range(days):
        nxt = pd.DataFrame({"date": [hist["date"].iloc[-1] + pd.Timedelta(days=1)],
                            "sales": [np.nan], "promo": [0]})
        row = build_features(pd.concat([hist, nxt], ignore_index=True)).iloc[[-1]][FEATS]
        nxt["sales"] = float(model.predict(row)[0] + row["roll28"].iloc[0])
        hist = pd.concat([hist, nxt], ignore_index=True)
    return hist.tail(days)


def main():
    raw = generate_data()                         # or: pd.read_csv("sales.csv", parse_dates=["date"])
    df = build_features(raw).dropna().reset_index(drop=True)
    train, test = df.iloc[:-TEST_DAYS], df.iloc[-TEST_DAYS:]

    model = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05,
                                      max_depth=3, subsample=0.8, random_state=42)
    # Trees cannot extrapolate a rising trend, so the model learns the deviation
    # from the recent 28-day level and we add that level back afterwards.
    model.fit(train[FEATS], train["sales"] - train["roll28"])
    pred = model.predict(test[FEATS]) + test["roll28"].values

    mae = mean_absolute_error(test["sales"], pred)
    mape = np.mean(np.abs((test["sales"] - pred) / test["sales"])) * 100
    base = mean_absolute_error(test["sales"], test["lag_7"])
    print(f"Gradient Boosting  MAE: {mae:.2f}  MAPE: {mape:.2f}%")
    print(f"Seasonal-naive     MAE: {base:.2f}   (same day last week)")
    print(f"Improvement over baseline: {(1 - mae / base) * 100:.1f}%")

    fut = forecast(model, raw)
    print(f"\nNext {HORIZON} days forecast (first 7):")
    print(fut[["date", "sales"]].head(7).assign(sales=lambda d: d["sales"].round(1))
          .to_string(index=False))

    fig, ax = plt.subplots(2, 1, figsize=(13, 8))
    ax[0].plot(test["date"], test["sales"], label="Actual")
    ax[0].plot(test["date"], pred, label="Predicted", alpha=0.85)
    ax[0].set_title(f"Last {TEST_DAYS} days: actual vs predicted (MAPE {mape:.1f}%)")
    ax[0].legend()
    ax[1].plot(raw["date"].tail(120), raw["sales"].tail(120), label="History")
    ax[1].plot(fut["date"], fut["sales"], color="tomato", label=f"{HORIZON}-day forecast")
    ax[1].set_title("Recursive forecast")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(OUT / "results.png", dpi=110)
    print("Saved results.png")


if __name__ == "__main__":
    main()
