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
    # Parameters from Pine Script
    fibLevel = 0.618
    lookback = 20

    # Calculate highest high and lowest low over lookback period
    highLevel = df['high'].rolling(window=lookback, min_periods=lookback).max()
    lowLevel = df['low'].rolling(window=lookback, min_periods=lookback).min()

    # Calculate Fibonacci retracement level
    fibRetracementLevel = lowLevel + (highLevel - lowLevel) * fibLevel

    # Entry condition: close <= fibRetracementLevel
    entry_condition = (df['close'] <= fibRetracementLevel) & fibRetracementLevel.notna()

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if entry_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_price_guess = float(df['close'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()

            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price_guess,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price_guess,
                'raw_price_b': entry_price_guess
            })
            trade_num += 1

    return entries