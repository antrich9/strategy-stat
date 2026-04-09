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
    # Parameters from original script
    fast_length = 50
    slow_length = 200
    session_long = '0700-0959'
    session_short = '1200-1459'
    gmt_offset = 1  # GMT+1

    entries = []
    trade_num = 1

    # Calculate EMAs
    fast_ema = df['close'].ewm(span=fast_length, adjust=False).mean()
    slow_ema = df['close'].ewm(span=slow_length, adjust=False).mean()

    # Crossover/crossunder: need previous bar to compare
    fast_above_slow = fast_ema > slow_ema
    crossover = (fast_above_slow) & (fast_ema.shift(1) <= slow_ema.shift(1))
    crossunder = (~fast_above_slow) & (fast_ema.shift(1) >= slow_ema.shift(1))

    # Previous day high/low using rolling 2-day window (shift to get prior day)
    prev_day_high = df['high'].rolling