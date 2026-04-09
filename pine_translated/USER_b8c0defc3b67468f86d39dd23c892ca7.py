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
    length = 5
    use_close_candle = False
    tradetype = "Long and Short"

    results = []
    trade_num = 1

    dir_up = False
    last_low = df['high'].iloc[0] * 100
    last_high = 0.0
    time_low = 0
    time_high = 0

    for i in range(len(df)):
        if i < length * 2 + 1:
            h = df['high'].iloc[:i+1].max()
            l = df['low'].iloc[:i+1].min()
        else:
            h = df['high'].iloc[i - length * 2:i + 1].max()
            l = df['low'].iloc[i - length * 2:i + 1].min()

        is_min = (l == df['low'].iloc[i])
        is_max = (h == df['high'].iloc[i])

        recent_touch = False
        for j in range(1, 11):
            if i + j + 1 < len(df):
                if (df['low'].iloc[i + j] <= last_low and df['low'].iloc[i + j + 1] > last_low) or (df['high'].iloc[i + j] >= last_high and df['high'].iloc[i + j + 1] < last_high):
                    recent_touch = True
                    break

        if dir_up:
            if is_min and df['low'].iloc[i] < last_low:
                last_low = df['low'].iloc[i]
                time_low = i
            if is_max and df['high'].iloc[i] > last_low:
                last_high = df['high'].iloc[i]
                time_high = i
                dir_up = False
        else:
            if is_max and df['high'].iloc[i] > last_high:
                last_high = df['high'].iloc[i]
                time_high = i
            if is_min and df['low'].iloc[i] < last_high:
                last_low = df['low'].iloc[i]
                time_low = i
                dir_up = True
                if is_max and df['high'].iloc[i] > last_low:
                    last_high = df['high'].iloc[i]
                    time_high = i
                    dir_up = False

        source_long = df['close'].iloc[i] if use_close_candle else df['high'].iloc[i]
        source_short = df['close'].iloc[i] if use_close_candle else df['low'].iloc[i]
        prev_high = df['high'].iloc[i - 1] if i > 0 else df['high'].iloc[i]
        prev_low = df['low'].iloc[i - 1] if i > 0 else df['low'].iloc[i]

        long_condition = source_long >= last_high and prev_high < last_high and not recent_touch and (tradetype == "Long and Short" or tradetype == "Long")
        short_condition = source_short <= last_low and prev_low > last_low and not recent_touch and (tradetype == "Long and Short" or tradetype == "Short")

        if long_condition and not short_condition:
            entry_price = df['close'].iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            results.append({
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

    return results