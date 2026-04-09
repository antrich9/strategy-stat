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
    high = df['high']
    low = df['low']

    # shifted series for swing detection
    high_s1 = high.shift(1)
    high_s2 = high.shift(2)
    high_s3 = high.shift(3)
    high_s4 = high.shift(4)

    low_s1 = low.shift(1)
    low_s2 = low.shift(2)
    low_s3 = low.shift(3)
    low_s4 = low.shift(4)

    # swing high: high[1] < high[2] and high[3] < high[2] and high[4] < high[2]
    is_swing_high = (high_s1 < high_s2) & (high_s3 < high_s2) & (high_s4 < high_s2)
    # swing low: low[1] > low[2] and low[3] > low[2] and low[4] > low[2]
    is_swing_low = (low_s1 > low_s2) & (low_s3 > low_s2) & (low_s4 > low_s2)

    entries = []
    trade_num = 1
    in_position = False

    for i in range(len(df)):
        # skip bars where swing indicators are not yet defined
        if pd.isna(is_swing_high.iloc[i]) or pd.isna(is_swing_low.iloc[i]):
            continue

        if not in_position and is_swing_low.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            in_position = True
        elif not in_position and is_swing_high.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            in_position = True

    return entries