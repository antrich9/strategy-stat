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

    if len(df) < 20:
        return entries

    # Time window: London sessions (07:45-09:45 and 13:45-15:45 UTC)
    # Since df doesn't have timezone info, assuming times are already UTC
    df_temp = df.copy()
    df_temp['hour'] = df_temp['time'].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc).hour)
    df_temp['minute'] = df_temp['time'].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc).minute)
    df_temp['time_minutes'] = df_temp['hour'] * 60 + df_temp['minute']
    
    morning_window = (df_temp['time_minutes'] >= 7 * 60 + 45) & (df_temp['time_minutes'] < 9 * 60 + 45)
    afternoon_window = (df_temp['time_minutes'] >= 13 * 60 + 45) & (df_temp['time_minutes'] < 15 * 60 + 45)
    in_trading_window = morning_window | afternoon_window

    # This Pine Script does not contain any strategy.entry() calls.
    # It is a visualization/indicator script that plots FVGs, swing points, and tables.
    # No entry logic exists to convert.

    return entries