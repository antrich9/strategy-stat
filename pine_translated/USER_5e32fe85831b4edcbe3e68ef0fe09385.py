import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    df columns: time(int unix ts), open(float), high(float), low(float), close(float), volume(float)
    Rows sorted ascending by time (oldest first). Index is 0-based int.

    Returns list of dicts:
    [{'trade_num': int, 'direction': 'long' or 'short',
      'entry_ts': int, 'entry_time': str,
      'entry_price_guess': float,
      'exit_ts': 0, 'exit_time': '', 'exit_price_guess': 0.0,
      'raw_price_a': float, 'raw_price_b': float}]
    """
    # ------------------------------------------------------------------
    # Helper: ALMA (Arnaud Legoux Moving Average)
    # ------------------------------------------------------------------
    def alma(series: pd.Series, length: int, offset: float = 0.85, sigma: float = 6.0) -> pd.Series:
        if length <= 0:
            return series
        if sigma == 0:
            return series.rolling(length).mean()
        k = np.arange(length)
        m = offset * (length - 1)
        w = np.exp(-((k - m) ** 2) / (2 * sigma ** 2))
        w_norm = w / w.sum()

        def _alma_impl(x):
            # x is a numpy array of length 'length', oldest to newest
            return np.dot(x, w_norm)

        return series.rolling(length).apply(_alma_impl, raw=True)

    # ------------------------------------------------------------------
    # Helper: GD (used for T3)
    # ------------------------------------------------------------------
    def gd(src: pd.Series, length: int, factor: float) -> pd.Series:
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 * (1 + factor) - ema2 * factor

    # ------------------------------------------------------------------
    # Inputs (hard‑coded values from the original Pine Script)
    # ------------------------------------------------------------------
    # Trendilo
    length_trendilo = 50
    alma_offset = 0.85
    alma_sigma = 6.0
    bmult = 1.0
    smooth_trendilo = 1

    # WAE
    sensitivity = 150
    fast_length = 20
    slow_length = 40
    channel_length = 20
    bb_mult = 2.0

    # T3
    use_t3 = True
    cross_t3 = True