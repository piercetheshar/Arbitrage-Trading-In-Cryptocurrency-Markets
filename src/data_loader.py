# src/data_loader.py

import ccxt
import pandas as pd


def fetch_all_data(symbol: str,
                   timeframe: str = "1h",
                   since_iso: str = "2019-08-01T00:00:00Z",
                   limit: int = 1000) -> pd.Series:
    """
    Fetch OHLCV data from BinanceUS for a single symbol and return
    a pandas Series of close prices indexed by timestamp.
    """
    exchange = ccxt.binanceus()

    since = exchange.parse8601(since_iso)
    all_ohlcv = []

    while True:
        ohlcv = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            since=since,
            limit=limit
        )
        if not ohlcv:
            break

        all_ohlcv += ohlcv

        # advance by one bar
        since = ohlcv[-1][0] + exchange.parse_timeframe(timeframe) * 1000

    df = pd.DataFrame(all_ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")

    # Return just the close price as a Series
    return df.set_index("ts")["c"].rename(symbol)


def load_pair(
    symbol_y: str = "DOT/USDT",
    symbol_x: str = "ADA/USDT",
    timeframe: str = "1h",
    since_iso: str = "2019-08-01T00:00:00Z",
) -> pd.DataFrame:
    """
    Fetch two crypto series and align them on their common timestamps.
    Returns a DataFrame with columns Close_Y and Close_X.
    """

    series_y = fetch_all_data(symbol_y, timeframe, since_iso)
    series_x = fetch_all_data(symbol_x, timeframe, since_iso)

    common_idx = series_y.index.intersection(series_x.index)

    data = pd.DataFrame(
        {
            "Close_Y": series_y.loc[common_idx],
            "Close_X": series_x.loc[common_idx],
        }
    )

    print("Merged shape:", data.shape)
    print("Date range:", data.index.min(), "→", data.index.max())

    return data