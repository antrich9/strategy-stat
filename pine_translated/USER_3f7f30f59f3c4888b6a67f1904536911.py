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
    # Period for breakout detection (rolling high/low)
    period = 20

    # Compute rolling high and low over the specified period
    rolling_high = df['high'].rolling(window=period).max()
    rolling_low = df['low'].rolling(window=period).min()

    # Entry conditions: price crosses above the rolling high (long)
    # or crosses below the rolling low (short)
    long_entry = (df['close'] > rolling_high) & (df['close'].shift(1) <= rolling_high.shift(1))
    short_entry = (df['close'] < rolling_low) & (df['close'].shift(1) >= rolling_low.shift(1))

    entries = []
    trade_num = 1

    # Iterate over the DataFrame starting after the first `period` bars
    for i in range(period, len(df)):
        # Skip bars where rolling indicators are not yet defined
        if pd.isna(rolling_high.iloc[i]) or pd.isna(rolling_low.iloc[i]):
            continue

        if long_entry.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif short_entry.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1

    return entries