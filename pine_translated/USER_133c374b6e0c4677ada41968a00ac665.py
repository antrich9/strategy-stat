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
    # compute EMAs
    ema9 = df['close'].ewm(span=9, adjust=False).mean()
    ema18 = df['close'].ewm(span=18, adjust=False).mean()

    # condition series
    cond_long = (df['close'] > ema9) & (df['close'] > ema18)
    cond_short = (df['close'] < ema9) & (df['close'] < ema18)

    entries = []
    trade_num = 1

    # iterate from index 1 to avoid NaN issues and to detect crossovers
    for i in range(1, len(df)):
        # skip if any required indicator is NaN
        if pd.isna(ema9.iloc[i]) or pd.isna(ema18.iloc[i]):
            continue
        # long entry: condition true now and was false previous bar
        if cond_long.iloc[i] and not cond_long.iloc[i-1]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
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
        # short entry: condition true now and was false previous bar
        if cond_short.iloc[i] and not cond_short.iloc[i-1]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
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