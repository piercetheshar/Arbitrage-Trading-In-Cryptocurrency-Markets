# src/signals.py

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from pykalman import KalmanFilter


def johansen_test(
    prices: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1,
    significance_level: float = 0.05,
):
    """
    Run Johansen cointegration test on a 2-column prices DataFrame.

    Returns:
        cointegrated (bool),
        trace_stat (float),
        crit_value (float)
    """
    if prices.shape[1] != 2:
        raise ValueError("prices must have exactly 2 columns")

    result = coint_johansen(prices, det_order=det_order, k_ar_diff=k_ar_diff)

    # result.cvt[0] = [critical 90%, 95%, 99%]
    level_to_idx = {0.10: 0, 0.05: 1, 0.01: 2}
    idx = level_to_idx.get(significance_level, 1)

    trace_stat = result.lr1[0]
    crit_value = result.cvt[0, idx]
    cointegrated = trace_stat > crit_value

    print(
        f"Johansen Test: Trace Statistic = {trace_stat:.2f}, "
        f"Critical Value ({int((1 - significance_level) * 100)}%) = {crit_value:.2f}"
    )
    print("Result:", "Cointegrated" if cointegrated else "Not cointegrated")

    return cointegrated, trace_stat, crit_value


def kalman_filter(y: pd.Series, x: pd.Series):
    """
    Use a Kalman Filter to estimate a time-varying hedge ratio and intercept
    between y (dependent) and x (independent) series.

    Returns:
        hedge_ratio (np.ndarray),
        intercept (np.ndarray)
    """
    # Make sure they are aligned
    y, x = y.align(x, join="inner")

    delta = 1e-5
    trans_cov = delta / (1 - delta) * np.eye(2)

    # Observation matrix: each observation is [x_t, 1]
    obs_mat = np.vstack([x.values, np.ones(len(x))]).T[:, np.newaxis]

    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=[0, 0],
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=1.0,
        transition_covariance=trans_cov,
    )

    state_means, _ = kf.filter(y.values)
    hedge_ratio = state_means[:, 0]
    intercept = state_means[:, 1]

    return hedge_ratio, intercept


def compute_spread(
    data: pd.DataFrame,
    y_col: str = "Close_Y",
    x_col: str = "Close_X",
):
    """
    Compute the Kalman-filter-based spread: Y - (hr * X + intercept).

    Adds a 'Spread' column to `data` and returns:
        data, hedge_ratio, intercept
    """
    hedge_ratio, intercept = kalman_filter(data[y_col], data[x_col])
    spread = data[y_col].values - (hedge_ratio * data[x_col].values + intercept)

    data = data.copy()
    data["Spread"] = spread

    return data, hedge_ratio, intercept


def calculate_zscore(spread: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate rolling Z-score for a spread series:
        Z = (spread - rolling_mean) / rolling_std
    """
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    zscore = (spread - rolling_mean) / rolling_std
    return zscore