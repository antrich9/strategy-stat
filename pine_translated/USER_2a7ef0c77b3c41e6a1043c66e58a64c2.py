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

    # Calculate threshold (auto=false case: use thresholdPer/100)
    threshold = 0 / 100  # thresholdPer input is 0

    # Calculate FVG conditions
    # Bullish FVG: low > high[2] and close[1] > high[2] and gap > threshold
    # Bearish FVG: high < low[2] and close[1] < low[2] and gap > threshold

    # Need at least 2 bars of lookback
    if len(df) < 3:
        return entries

    # Calculate shifted values
    high_2 = df['high'].shift(2)
    low_2 = df['low'].shift(2)
    close_1 = df['close'].shift(1)

    # Bullish FVG condition
    bull_fvg = (
        (df['low'] > high_2) & 
        (close_1 > high_2) & 
        ((df['low'] - high_2) / high_2 > threshold)
    )

    # Bearish FVG condition
    bear_fvg = (
        (df['high'] < low_2) & 
        (close_1 < low_2) & 
        ((low_2 - df['high']) / df['high'] > threshold)
    )

    # Iterate and create entries
    for i in range(2, len(df)):
        if pd.isna(high_2.iloc[i]) or pd.isna(low_2.iloc[i]) or pd.isna(close_1.iloc[i]):
            continue

        direction = None
        entry_price = df['close'].iloc[i]
        entry_ts = df['time'].iloc[i]
        entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()

        if bull_fvg.iloc[i]:
            direction = 'long'
        elif bear_fvg.iloc[i]:
            direction = 'short'

        if direction:
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': int(entry_ts),
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1

    return entries