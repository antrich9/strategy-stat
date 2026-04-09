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
    # Parameters controlling the strategy
    max_positions_per_day = 2
    wait_bars = 5

    # Detect high‑impact news. In the original script this comes from an external
    # library. Here we assume a boolean column 'high_impact' exists in the DataFrame.
    # If the column is missing we default to False (i.e. no entries will be generated).
    if 'high_impact' in df.columns:
        high_impact = df['high_impact'].astype(bool)
    else:
        high_impact = pd.Series(False, index=df.index)

    entries = []
    trade_num = 1

    # State variables
    positions_today = 0
    waiting_for_entry = False
    wait_count = 0
    prev_date = None

    for i in range(len(df)):
        row = df.iloc[i]
        ts = row['time']
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        date = dt.date()

        # New day → reset daily position counter and waiting state
        if date != prev_date:
            positions_today = 0
            waiting_for_entry = False
            wait_count = 0
            prev_date = date

        # If a high‑impact news bar is detected and we can open a new position,
        # start the waiting period.
        if high_impact.iloc[i] and positions_today < max_positions_per_day:
            waiting_for_entry = True
            wait_count = 0

        # Count bars while waiting and fire the entry when the required
        # number of bars (wait_bars) has been reached.
        if waiting_for_entry:
            wait_count += 1
            if wait_count == wait_bars:
                entry_price = row['close']
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(ts),
                    'entry_time': dt.isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price,
                })
                trade_num += 1
                positions_today += 1
                waiting_for_entry = False
                wait_count = 0

    return entries