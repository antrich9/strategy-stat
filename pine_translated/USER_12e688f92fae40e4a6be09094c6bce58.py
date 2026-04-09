import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

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
    # DST detection for UK (Europe/London)
    def is_dst(dt: datetime) -> bool:
        """Return True if the given naive UTC datetime is within UK DST."""
        year = dt.year
        # Last Sunday of March
        march_31 = datetime(year, 3, 31)
        march_wday = march_31.weekday()  # Monday=0, Sunday=6
        days_to