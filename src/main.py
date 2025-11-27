# src/main.py

import matplotlib.pyplot as plt

from data_loader import load_pair
from signals import johansen_test, compute_spread, calculate_zscore
from backtest import backtest_trading


def run():
    # 1) Load data (you can change the pair here)
    data = load_pair(
        symbol_y="DOT/USDT",
        symbol_x="ADA/USDT",
        timeframe="1h",
        since_iso="2019-08-01T00:00:00Z",
    )

    # 2) Johansen cointegration test
    cointegrated, trace_stat, crit_value = johansen_test(
        data[["Close_Y", "Close_X"]]
    )

    if not cointegrated:
        print("WARNING: Pair is NOT cointegrated – strategy may not be valid.")
    else:
        print("Proceeding with Kalman filter & backtest...")

    # 3) Kalman-based hedge ratio & spread
    data, hedge_ratio, intercept = compute_spread(
        data,
        y_col="Close_Y",
        x_col="Close_X",
    )

    # 4) Z-score of the spread
    data["Zscore"] = calculate_zscore(data["Spread"], window=20)

    # 5) (Optional) visualize Z-score
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["Zscore"], label="Z-Score")
    plt.axhline(2, linestyle="--", label="Upper Threshold (2)")
    plt.axhline(-2, linestyle="--", label="Lower Threshold (-2)")
    plt.axhline(0, linestyle="--", label="Mean (0)")
    plt.title("Rolling Z-Score of Spread")
    plt.xlabel("Date")
    plt.ylabel("Z-Score")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 6) Backtest
    trade_returns, total_profit = backtest_trading(
        data,
        hedge_ratio=hedge_ratio,
        y_col="Close_Y",
        x_col="Close_X",
        z_col="Zscore",
        z_entry=2.0,
        z_exit=0.0,
    )

    print("Number of closed trades:", len(trade_returns))


if __name__ == "__main__":
    run()