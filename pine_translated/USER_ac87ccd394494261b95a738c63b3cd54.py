import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']

    # Hull MA (not used in entry logic but computed for completeness)
    length_hull = 9
    half_len = int(length_hull / 2)
    sqrt_len = int(np.sqrt(length_hull))

    def wma(series, window):
        weights = np.arange(1, window + 1)
        def _wm(x):
            return np.dot(x, weights) / weights.sum()
        return series.rolling(window).apply(_wm, raw=True)

    hull_half = wma(close, half_len)
    hull_full = wma(close, length_hull)
    hull_raw = 2 * hull_half - hull_full
    hullma = wma(hull_raw, sqrt_len)

    # T3
    length_t3 = 5
    factor_t3 = 0.7

    def gd(src, length, factor):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 * (1 + factor) - ema2 * factor

    t3 = gd(close, length_t3, factor_t3)
    t3 = gd(t3, length_t3, factor_t3)
    t3 = gd(t3, length_t3, factor_t3)
    t3_rising = t3 > t3.shift(1)

    # Stiffness
    ma_len = 100
    stiff_len = 60
    stiff_smooth = 3
    threshold = 90

    sma = close.rolling(ma_len).mean()
    std = close.rolling(ma_len).std()
    bound = sma - 0.2 * std
    sum_above = (close > bound).rolling(stiff_len).sum()
    stiffness = (sum_above * 100.0 / stiff_len).ewm(span=stiff_smooth, adjust=False).mean()

    # Entry conditions
    basic_long = t3_rising & (close > t3)
    t3_long_cross = basic_long & ~basic_long.shift(1).fillna(False)
    long_cond = t3_long_cross & (stiffness > threshold)

    basic_short = (~t3_rising) & (close < t3)
    t3_short_cross = basic_short & ~basic_short.shift(1).fillna(False)
    short_cond = t3_short_cross & (stiffness < threshold)

    # Generate entry list
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if long_cond.iloc[i]:
            ts_val = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts_val,
                'entry_time': datetime.fromtimestamp(ts_val, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif short_cond.iloc[i]:
            ts_val = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts_val,
                'entry_time': datetime.fromtimestamp(ts_val, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    return entries