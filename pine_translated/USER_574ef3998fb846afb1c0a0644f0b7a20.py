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
    # Constants from the original Pine Script
    pip_size = 0.0002
    fast_length = 8
    medium_length = 20
    slow_length = 50
    atr_length = 7

    # Fixed date range (UTC)
    start_ts = 1577836800   # 2020-01-01 00:00:00 UTC
    end_ts = 1609459199     # 2020-12-31 23:59:59 UTC

    # Price series
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']

    # Exponential Moving Averages (EMA)
    fast_ema = close.ewm(span=fast_length, adjust=False).mean()
    medium_ema = close.ewm(span=medium_length, adjust=False).mean()
    slow_ema = close.ewm(span=slow_length, adjust=False).mean()

    # True Range and Wilder ATR
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2.fillna(0), tr3.fillna(0)], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / atr_length, adjust=False).mean()

    # Price‑action threshold
    range_bar = high - low
    upper_threshold = high - 0.3 * range_bar

    # Bullish candle condition
    bullish_candle = (close > upper_threshold) & (open_ > upper_threshold)

    # EMA alignment
    long_emas_aligned = (fast_ema > medium_ema) & (medium_ema > slow_ema)

    # Entry condition
    cond = (
        bullish_candle
        & long_emas_aligned
        & ((close <= fast_ema) | (low <= fast_ema))
    )

    # Date‑range filter
    in_date = (df['time'] >= start_ts) & (df['time'] <= end_ts)
    cond = cond & in_date

    # Mask out rows where any required indicator is NaN
    indicator_valid = ~fast_ema.isna() & ~medium_ema.isna() & ~slow_ema.isna()
    cond = cond & indicator_valid
    cond = cond.fillna(False)

    # Build entry list
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = float(df['high'].iloc[i] + pip_size)
            entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
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