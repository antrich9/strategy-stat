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
    # Detect bullish and bearish Fair Value Gaps (FVG)
    bull_fvg = (df['low'] > df['high'].shift(2)) & (df['close'].shift(1) > df['high'].shift(2)) & (df['open'].shift(2) < df['close'].shift(2)) & (df['open'].shift(1) < df['close'].shift(1)) & (df['open'] < df['close'])
    bear_fvg = (df['high'] < df['low'].shift(2)) & (df['close'].shift(1) < df['low'].shift(2)) & (df['open'].shift(2) > df['close'].shift(2)) & (df['open'].shift(1) > df['close'].shift(1)) & (df['open'] > df['close'])

    entries = []
    trade_num = 1

    # Start from index 2 to avoid NaNs from shifts
    for i in range(2, len(df)):
        if bull_fvg.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif bear_fvg.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries