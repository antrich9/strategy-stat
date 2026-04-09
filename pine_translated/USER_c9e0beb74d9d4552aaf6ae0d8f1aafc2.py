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
    # Ensure required columns exist
    required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("DataFrame must contain columns: time, open, high, low, close, volume")

    # Work on a copy to avoid modifying original
    df = df.copy()

    # Convert time to datetime (assuming Unix timestamp in seconds)
    # Use UTC for processing, then convert to desired timezone
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)

    # Assume time zone input is GMT+1 (as per default in Pine Script)
    # In tz database, GMT+1 is represented as 'Etc/GMT-1' (sign is reversed)
    try:
        df['datetime'] = df['datetime'].dt.tz_convert('Etc/GMT-1')
    except Exception:
        # If conversion fails, keep in UTC
        pass

    # Extract date and time components
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt