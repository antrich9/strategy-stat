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
    # make a copy to avoid mutating input
    df = df.copy()
    # ensure sorted by time
    df = df.sort_values('time').reset_index(drop=True)

    open_s = df['open']
    high_s = df['high']
    low_s = df['low']
    close_s = df['close']

    # body size and body percent
    body = (close_s - open_s).abs()
    hl_range = high_s - low_s
    # avoid division by zero
    body_pct = np.where(hl_range == 0, 0.0, body * 100.0 / hl_range)

    # candle direction
    VVE_0 = close_s > open_s
    VRE_0 = close_s < open_s

    # elephant candle body percent threshold
    PDCM = 70