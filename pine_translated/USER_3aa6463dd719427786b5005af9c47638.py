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
    # ZigZag period (default from Pine script)
    prd = 2
    left_size = prd - 1   # number of bars to look back
    right_size = prd - 1  # number of bars to look forward

    high = df['high']
    low = df['low']
    close = df['close']

    n = len(df)

    # Initialise pivot high/low series
    ph = pd.Series(False, index=df.index)
    pl = pd.Series(False, index=df.index)

    high_arr = high.values
    low_arr = low.values

    # Compute pivot points using a simple local extrema check
    for i in range(left_size, n - right_size):
        # left window (excluding current bar)
        left_high_max = high_arr[i - left_size:i].max()
        left_low_min = low_arr[i - left_size:i].min()
        # right window (excluding current bar)
        right_high_max = high_arr[i + 1:i + right_size + 1].max()
        right_low_min = low_arr[i + 1:i + right_size + 1].min()

        if high_arr[i] > left_high_max and high_arr[i] > right_high_max:
            ph.iloc[i] = True
        if low_arr[i] < left_low_min and low_arr[i] < right_low_min:
            pl.iloc[i] = True

    # Determine direction series (1 = long, -1 = short, 0 = undefined)
    dir_seq = pd.Series(0, index=df.index)
    prev_dir = 0
    for i in range(left_size, n):
        if ph.iloc[i] and not pl.iloc[i]:
            curr_dir = 1
        elif pl.iloc[i] and not ph.iloc[i]:
            curr_dir = -1
        else:
            curr_dir = prev_dir
        dir_seq.iloc[i] = curr_dir
        prev_dir = curr_dir

    # Detect changes in direction
    dir_changed = dir_seq.diff() != 0
    dir_changed.iloc[0] = False  # first bar cannot be a change

    # Long and short entry signals
    long_entry = (dir_seq == 1) & dir_changed
    short_entry = (dir_seq == -1) & dir_changed

    # Build entry list
    entries = []
    trade_num = 1

    for i in df.index:
        if long_entry.iloc[i] or short_entry.iloc[i]:
            direction = 'long' if long_entry.iloc[i] else 'short'
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
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