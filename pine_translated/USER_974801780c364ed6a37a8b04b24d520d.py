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
    # Compute EMAs (typical fast=9, slow=21)
    fast_ema = df['close'].ewm(span=9, adjust=False).mean()
    slow_ema = df['close'].ewm(span=21, adjust=False).mean()

    # Optional Fibonacci 0.618 level based on a rolling 50-bar swing high/low
    window = 50
    high_max = df['high'].rolling(window=window).max()
    low_min = df['low'].rolling(window=window).min()
    fib618 = low_min + (high_max - low_min) * 0.618

    # Build boolean series for entry conditions
    long_cond = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
    short_cond = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))

    entries = []
    trade_num = 1

    for i in range(1, len(df)):
        if pd.isna(fast_ema.iloc[i]) or pd.isna(slow_ema.iloc[i]) or pd.isna(fib618.iloc[i]):
            continue

        if long_cond.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif short_cond.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
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