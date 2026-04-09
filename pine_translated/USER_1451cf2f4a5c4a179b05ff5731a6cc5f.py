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
    # Settings
    inp11 = False  # Volume Filter
    inp21 = False  # ATR Filter
    inp31 = False  # Trend Filter
    
    atr_length1 = 20
    
    # Convert time to datetime with UTC, then to London timezone
    df = df.copy()
    df['dt_utc'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['dt_london'] = df['dt_utc'].dt.tz_convert('Europe/London')
    df['hour'] = df['dt_london'].dt.hour
    df['minute'] = df['dt_london'].dt.minute
    
    # Trading windows
    # Window 1: 07:00-11:45
    in_window1 = (df['hour'] == 7) | ((df['hour'] >= 8) & (df['hour'] < 11)) | ((df['hour'] == 11) & (df['minute'] <= 45))
    # Window 2: 14:00-14:45
    in_window2 = (df['hour'] == 14) & (df['minute'] <= 45)
    
    df['in_trading_window'] = in_window1 | in_window2