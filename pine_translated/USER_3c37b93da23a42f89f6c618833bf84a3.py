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
    auto = False
    thresholdPer = 0.0

    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert('Europe/London')
    df['hour'] = df['dt'].dt.hour
    df['minute'] = df['dt'].dt.minute

    window1 = ((df['hour'] == 7) & (df['minute'] >= 45)) | \
              ((df['hour'] >= 8) & (df['hour'] <= 10)) | \
              ((df['hour'] == 11) & (df['minute'] <= 45))
    window2 = (df['hour'] == 14) & (df['minute'] <= 45)
    in_trading_window = window1 | window2

    if auto:
        threshold = ((df['high'] - df['low']) / df['low']).expanding().mean()
    else:
        threshold = thresholdPer / 100

    bull_fvg = (df['low'] > df['high'].shift(2)) & \
               (df['close'].shift(1) > df['high'].shift(2)) & \
               ((df['low'] - df['high'].shift(2)) / df['high'].shift(2) > threshold)

    bear_fvg = (df['high'] < df['low'].shift(2)) & \
               (df['close'].shift(1) < df['low'].shift(2)) & \
               ((df['low'].shift(2) - df['high']) / df['high'] > threshold)

    lastFVG = pd.Series(0, index=df.index)
    consecutiveBullCount = 0
    consecutiveBearCount = 0

    entries = []
    trade_num