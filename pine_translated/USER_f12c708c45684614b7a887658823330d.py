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
    # Ensure numeric types
    df = df.copy()
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['time'] = pd.to_numeric(df['time'], errors='coerce')

    # ---- Hull MA parameters ----
    length_hull = 9
    half_hull = length_hull // 2
    sqrt_hull = int(np.sqrt(length_hull))

    # Weighted Moving Average (WMA)
    def wma(series, length):
        if length <= 0:
            return series * 0
        weights = np.arange(1, length + 1, dtype=float)
        def _wma(x):
            return np.dot(x, weights) / weights.sum()
        return series.rolling(length).apply(_wma, raw=True)

    # Hull MA
    wma_half = wma(df['close'], half_hull)
    wma_full = wma(df['close'], length_hull)
    hull = wma(2 * wma_half - wma_full, sqrt_hull)

    # Hull signal: 1 if rising, -1 otherwise
    sig_hull = pd.Series(np.where(hull > hull.shift(1), 1, -1), index=df.index)

    # ---- T3 parameters ----
    length_t3 = 5
    factor_t3 = 0.7

    # Generalized EMA (gd)
    def gd(src, length, factor):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 * (1 + factor) - ema2 * factor

    # Triple gd to obtain T3
    def compute_t3(src, length, factor):
        g1 = gd(src, length, factor)
        g2 = gd(g1, length, factor)
        g3 = gd(g2, length, factor)
        return g3

    t3 = compute_t3(df['close'], length_t3, factor_t3)

    # T3 signal: 1 if rising, -1 otherwise
    t3_signal = pd.Series(np.where(t3 > t3.shift(1), 1, -1), index=df.index)

    close = df['close']

    # ---- Hull MA based conditions ----
    signal_hull_long = (sig_hull > 0) & (close > hull)
    signal_hull_short = (sig_hull < 0) & (close < hull)

    # ---- T3 based conditions ----
    basic_long_cond = (t3_signal > 0) & (close > t3)
    basic_short_cond = (t3_signal < 0) & (close < t3)

    # Cross confirmation (crossT3 = True)
    t3_long = basic_long_cond
    t3_long_cross = t3_long & (~t3_long.shift(1).fillna(False).astype(bool))

    t3_short = basic_short_cond
    t3_short_cross = t3_short & (~t3_short.shift(1).fillna(False).astype(bool))

    # inverseT3 = False
    t3_long_final = t3_long_cross
    t3_short_final = t3_short_cross

    # ---- Final entry conditions ----
    long_condition = signal_hull_long & t3_long_final
    short_condition = signal_hull_short & t3_short_final

    # ---- Build entry list ----
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if long_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    return entries