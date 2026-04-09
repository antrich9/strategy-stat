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
    close = df['close']

    # ---------- Hull MA ----------
    length_hull = 9
    half_len = int(length_hull / 2)
    sqrt_len = int(np.sqrt(length_hull))

    def wma(series, window):
        weights = np.arange(1, window + 1)
        def weighted_avg(x):
            return np.dot(x, weights) / weights.sum()
        return series.rolling(window=window).apply(weighted_avg, raw=True)

    wma_half = wma(close, half_len)
    wma_full = wma(close, length_hull)
    raw_hull = 2 * wma_half - wma_full
    hullma = wma(raw_hull, sqrt_len)

    sig_hull = pd.Series(np.where(hullma > hullma.shift(1), 1, -1), index=df.index)
    signal_hull_long = (sig_hull > 0) & (close > hullma)

    # ---------- T3 ----------
    length_t3 = 5
    factor_t3 = 0.7

    def ema(series, span):
        return series.ewm(span=span, adjust=False).mean()

    def gd_t3(src, length, factor):
        ema1 = ema(src, length)
        ema2 = ema(ema1, length)
        return ema1 * (1 + factor) - ema2 * factor

    t3_1 = gd_t3(close, length_t3, factor_t3)
    t3_2 = gd_t3(t3_1, length_t3, factor_t3)
    t3 = gd_t3(t3_2, length_t3, factor_t3)

    t3_signals = pd.Series(np.where(t3 > t3.shift(1), 1, -1), index=df.index)

    basic_long_cond = (t3_signals > 0) & (close > t3)
    t3_signals_long = basic_long_cond

    prev_t3_long = t3_signals_long.shift(1).fillna(True)
    t3_signals_long_cross = (~prev_t3_long) & t3_signals_long
    t3_signals_long_final = t3_signals_long_cross

    # ---------- Stiffness ----------
    ma_len_stiff = 100
    stiff_len = 60
    stiff_smooth = 3
    threshold_stiff = 90

    sma_stiff = close.rolling(window=ma_len_stiff).mean()
    std_stiff = close.rolling(window=ma_len_stiff).std()
    bound_stiff = sma_stiff - 0.2 * std_stiff

    above_bound = (close > bound_stiff).astype(int)
    sum_above = above_bound.rolling(window=stiff_len).sum()
    stiffness_pct = sum_above * 100.0 / stiff_len
    stiffness = stiffness_pct.ewm(span=stiff_smooth, adjust=False).mean()

    stiffness_cond = stiffness > threshold_stiff

    # ---------- Combined entry ----------
    entry_condition = signal_hull_long & t3_signals_long_final & stiffness_cond
    entry_condition = entry_condition.fillna(False)

    # ---------- Generate entries ----------
    entries = []
    trade_num = 1

    for i in df.index:
        if entry_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries