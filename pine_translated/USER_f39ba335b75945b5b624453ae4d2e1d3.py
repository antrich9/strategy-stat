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

    open_arr = df['open']
    high_arr = df['high']
    low_arr = df['low']
    close_arr = df['close']

    # Bullish FVG: low[2] <= open[1] and high[0] >= close[1] and close[0] < low[1]
    bullish_fvg = (low_arr.shift(2) <= open_arr.shift(1)) & \
                   (high_arr >= close_arr.shift(1)) & \
                   (close_arr < low_arr.shift(1))

    # Bearish FVG: high[2] >= open[1] and low[0] <= close[1] and close[0] > high[1]
    bearish_fvg = (high_arr.shift(2) >= open_arr.shift(1)) & \
                   (low_arr <= close_arr.shift(1)) & \
                   (close_arr > high_arr.shift(1))

    n = len(df)
    for i in range(2, n):
        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        entry_price = close_arr.iloc[i]

        if bullish_fvg.iloc[i]:
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

        if bearish_fvg.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
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