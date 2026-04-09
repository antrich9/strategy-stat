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
    # Ensure DataFrame is sorted by time
    df = df.sort_values('time').reset_index(drop=True)

    # Compute EMAs (8, 20, 50) using Wilder's smoothing via ewm
    ema8 = df['close'].ewm(span=8, adjust=False).mean()
    ema20 = df['close'].ewm(span=20, adjust=False).mean()
    ema50 = df['close'].ewm(span=50, adjust=False).mean()

    # Identify crossover / crossunder
    # Long: EMA8 crosses above EMA20 while EMA20 > EMA50
    long_cond = (ema8 > ema20) & (ema8.shift(1) <= ema20.shift(1)) & (ema20 > ema50)
    # Short: EMA8 crosses below EMA20 while EMA20 < EMA50
    short_cond = (ema8 < ema20) & (ema8.shift(1) >= ema20.shift(1)) & (ema20 < ema50)

    entries = []
    trade_num = 1

    for i in range(1, len(df)):
        # Skip bars with NaN in any required indicator
        if (pd.isna(ema8.iloc[i]) or pd.isna(ema20.iloc[i]) or pd.isna(ema50.iloc[i])):
            continue

        if long_cond.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])

            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

        elif short_cond.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])

            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries