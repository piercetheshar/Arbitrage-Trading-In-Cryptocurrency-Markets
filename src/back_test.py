# src/backtest.py

import numpy as np
import pandas as pd


def backtest_trading(
    data: pd.DataFrame,
    hedge_ratio: np.ndarray,
    y_col: str = "Close_Y",
    x_col: str = "Close_X",
    z_col: str = "Zscore",
    z_entry: float = 2.0,
    z_exit: float = 0.0,
):
    """
    Very simple mean-reversion backtest:

    - When Z > z_entry: short Y, long X * hedge_ratio
    - When Z < -z_entry: long Y, short X * hedge_ratio
    - Exit when Z back to +/- z_exit.

    Returns:
        trade_returns (list of PnL per closed trade),
        total_profit (float)
    """
    data = data.copy()

    position = 0  # 0 = flat, 1 = long Y/short X, -1 = short Y/long X
    entry_price_y = None
    entry_price_x = None

    trade_returns = []

    for i, (ts, row) in enumerate(data.iterrows()):
        z = row[z_col]
        if not np.isfinite(z):
            continue

        price_y = row[y_col]
        price_x = row[x_col]
        hr = hedge_ratio[i]

        # ENTRY
        if position == 0:
            if z > z_entry:
                # Short Y, long X*hr
                position = -1
                entry_price_y = price_y
                entry_price_x = price_x
                print(f"{ts}: Enter SHORT {y_col}, LONG {x_col} (Z = {z:.2f})")

            elif z < -z_entry:
                # Long Y, short X*hr
                position = 1
                entry_price_y = price_y
                entry_price_x = price_x
                print(f"{ts}: Enter LONG {y_col}, SHORT {x_col} (Z = {z:.2f})")

        # EXIT
        else:
            exit_signal = (position == 1 and z >= z_exit) or (
                position == -1 and z <= z_exit
            )

            if exit_signal:
                if position == 1:
                    # Long Y, short X*hr
                    pnl = (price_y - entry_price_y) - hr * (price_x - entry_price_x)
                    trade_type = f"LONG {y_col}, SHORT {x_col}"
                else:
                    # Short Y, long X*hr
                    pnl = (entry_price_y - price_y) - hr * (entry_price_x - price_x)
                    trade_type = f"SHORT {y_col}, LONG {x_col}"

                trade_returns.append(pnl)
                print(f"{ts}: Exit {trade_type} with PnL: {pnl:.2f}")

                position = 0
                entry_price_y = None
                entry_price_x = None

    total_profit = float(np.nansum(trade_returns))
    print(f"\nTotal Profit from Backtesting: {total_profit:.2f}")

    return trade_returns, total_profit