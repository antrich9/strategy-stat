import pandas as pd
import numpy as np
from datetime import datetime, timezone
import math

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
    # Strategy parameters (matching Pine script defaults)
    use_hull = True
    use_color_hull = True
    length_hull = 9
    src_hull = df['close']

    use_t3 = True
    highlight_movements_t3 = True
    cross_t3 = True
    inverse_t3 = False
    length_t3 = 5
    factor_t3 = 0.7

    use_kalman = True
    q = 0.001
    r = 0.001

    # ---------- Helper: Weighted Moving Average ----------
    def wma(series: pd.Series, n: int) -> pd.Series:
        if n <= 0:
            return series * np.nan
        weights = np.arange(1, n + 1)
        def weighted_sum(x):
            return np.dot(x, weights) / weights.sum()
        return series.rolling(window=n, min_periods=n).apply(weighted_sum, raw=True)

    # ---------- Hull MA ----------
    half_len = int(length_hull / 2)
    sqrt_len = int(math.sqrt(length_hull))
    wma_half = wma(src_hull, half_len)
    wma_full = wma(src_hull, length_hull)
    hull_raw = 2 * wma_half - wma_full
    hull_ma = wma(hull_raw, sqrt_len)

    hull_up = hull_ma > hull_ma.shift(1)
    sig_hull = pd.Series(np.where(hull_up, 1, -1), index=hull_ma.index)

    if use_hull:
        if use_color_hull:
            cond_hull = (sig_hull > 0) & (df['close'] > hull_ma)
        else:
            cond_hull = df['close'] > hull_ma
    else:
        cond_hull = pd.Series(True, index=df.index)

    # ---------- T3 ----------
    def t3_func(series: pd.Series, length: int, factor: float) -> pd.Series:
        ema = series.ewm(span=length, adjust=False).mean()
        ema2 = ema.ewm(span=length, adjust=False).mean()
        gd = ema * (1 + factor) - ema2 * factor
        for _ in range(2):
            ema = gd.ewm(span=length, adjust=False).mean()
            ema2 = ema.ewm(span=length, adjust=False).mean()
            gd = ema * (1 + factor) - ema2 * factor
        return gd

    t3_series = t3_func(df['close'], length_t3, factor_t3)

    t3_up = t3_series > t3_series.shift(1)
    t3_signals = pd.Series(np.where(t3_up, 1, -1), index=t3_series.index)

    basic_long_cond = (t3_signals > 0) & (df['close'] > t3_series)

    if use_t3:
        if highlight_movements_t3:
            t3_long = basic_long_cond
        else:
            t3_long = df['close'] > t3_series
    else:
        t3_long = pd.Series(True, index=df.index)

    prev_t3_long = t3_long.shift(1).fillna(False)
    if cross_t3:
        t3_long_cross = (~prev_t3_long) & t3_long
    else:
        t3_long_cross = t3_long

    if inverse_t3:
        t3_long_final = ~t3_long_cross
    else:
        t3_long_final = t3_long_cross

    # ---------- Kalman Filter ----------
    def kalman_filter(series: pd.Series, q: float, r: float) -> pd.Series:
        n = len(series)
        x = np.nan
        p = 1.0
        kalman = np.full(n, np.nan)
        for i in range(n):
            src_val = series.iloc[i]
            if np.isnan(x):
                x = src_val
            x_pred = x
            p_pred = p + q
            k = p_pred / (p_pred + r)
            x = x_pred + k * (src_val - x_pred)
            p = (1 - k) * p_pred
            kalman[i] = x
        return pd.Series(kalman, index=series.index)

    kalman_price = kalman_filter(df['close'], q, r)

    if use_kalman:
        cond_kalman = df['close'] > kalman_price
    else:
        cond_kalman = pd.Series(True, index=df.index)

    # ---------- Combined Entry Condition ----------
    entry_cond = cond_hull & t3_long_final & cond_kalman

    # ---------- Generate Entries ----------
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if pd.isna(hull_ma.iloc[i]):
            continue
        if pd.isna(t3_series.iloc[i]):
            continue
        if pd.isna(kalman_price.iloc[i]):
            continue
        if not entry_cond.iloc[i]:
            continue

        ts = int(df['time'].iloc[i])
        entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        entry_price = df['close'].iloc[i]

        entries.append({
            'trade_num': trade_num,
            'direction': 'long',
            'entry_ts': ts,
            'entry_time': entry_time_str,
            'entry_price_guess': entry_price,
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': entry_price,
            'raw_price_b': entry_price
        })
        trade_num += 1

    return entries