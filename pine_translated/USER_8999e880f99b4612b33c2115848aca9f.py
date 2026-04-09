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
    # Shifted series for Pine Script references
    low2 = df['low'].shift(2)
    low1 = df['low'].shift(1)
    low0 = df['low']
    high2 = df['high'].shift(2)
    high1 = df['high'].shift(1)
    high0 = df['high']
    open1 = df['open'].shift(1)
    close1 = df['close'].shift(1)
    close0 = df['close']

    # Bearish breakaway FVG -> short entry
    short_cond = (
        (low2 <= open1) &
        (high0 >= close1) &
        (close0 < low1) &
        ((low2 - high0) > 0)
    )

    # Bullish breakaway FVG -> long entry
    long_cond = (
        (high2 >= open1) &
        (low0 <= close1) &
        (close0 > high1) &
        ((low0 - high2) > 0)
    )

    entries = []
    trade_num = 1

    for i in range(2, len(df)):
        if short_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif long_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

    return entries