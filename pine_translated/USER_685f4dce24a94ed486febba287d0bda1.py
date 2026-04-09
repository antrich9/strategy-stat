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
    # Ensure sorted by time
    df = df.sort_values('time').reset_index(drop=True)

    # Convert time to datetime (UTC)
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)

    # Extract date for daily grouping
    df['day'] = df['datetime'].dt.date

    # Compute daily high and low
    daily_hl = df.groupby('day')['high'].max().to_frame('high_day')
    daily_hl['low_day'] = df.groupby('day')['low'].min()

    # Shift to get previous day's high/low
    daily_hl['prev_day_high'] = daily_hl['high_day'].shift(1)
    daily_hl['prev_day_low'] = daily_hl['low_day'].shift(1)

    # Merge previous day high/low back to bars
    df = df.merge(daily_hl[['prev_day_high', 'prev_day_low']], on='day', how='left')

    # Fill NaNs for first day (if any) with forward fill
    df['prev_day_high'] = df['prev_day_high'].ffill()
    df['prev_day_low'] = df['prev_day_low'].ffill()

    # Determine sweep conditions
    df['sweep_high'] = df