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
    n = len(df)
    if n < 3:
        return []

    open_series = df['open']
    high_series = df['high']
    low_series = df['low']
    close_series = df['close']

    is_up = close_series > open_series
    is_down = close_series < open_series

    ob_up = (is_down.shift(1)) & (is_up) & (close_series > high_series.shift(1))
    ob_down = (is_up.shift(1)) & (is_down) & (close_series < low_series.shift(1))

    fvg_up = low_series > high_series.shift(2)
    fvg_down = high_series < low_series.shift(2)

    bfvg = (low_series > high_series.shift(2))
    sfvg = (high_series < low_series.shift(2))

    entries = []
    trade_num = 1

    for i in range(2, n):
        long_cond = (ob_up.iloc[i] and fvg_up.iloc[i]) or (bfvg.iloc[i])
        short_cond = (ob_down.iloc[i] and fvg_down.iloc[i]) or (sfvg.iloc[i])

        direction = None
        if long_cond:
            direction = 'long'
        elif short_cond:
            direction = 'short'

        if direction:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(close_series.iloc[i])

            entries.append({
                'trade_num': trade_num,
                'direction': direction,
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