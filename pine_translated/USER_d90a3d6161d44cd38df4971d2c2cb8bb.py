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
    entries = []
    trade_num = 1

    n = len(df)
    if n < 3:
        return entries

    close = df['close']
    open_vals = df['open']
    high = df['high']
    low = df['low']

    for i in range(2, n):
        is_up_curr = close.iloc[i] > open_vals.iloc[i]
        is_down_curr = close.iloc[i] < open_vals.iloc[i]
        is_down_prev = close.iloc[i-1] < open_vals.iloc[i-1]
        is_up_prev = close.iloc[i-1] > open_vals.iloc[i-1]

        close_above_prev_high = close.iloc[i] > high.iloc[i-1]
        close_below_prev_low = close.iloc[i] < low.iloc[i-1]

        fvg_up = low.iloc[i] > high.iloc[i-2]
        fvg_down = high.iloc[i] < low.iloc[i-2]

        if is_down_prev and is_up_curr and close_above_prev_high and fvg_up:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1

        if is_up_prev and is_down_curr and close_below_prev_low and fvg_down:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1

    return entries