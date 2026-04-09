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
    # Compute EMAs
    ema8 = df['close'].ewm(span=8, adjust=False).mean()
    ema20 = df['close'].ewm(span=20, adjust=False).mean()
    ema50 = df['close'].ewm(span=50, adjust=False).mean()

    # Crossover and crossunder detection
    crossover = (ema8 > ema20) & (ema8.shift(1).le(ema20.shift(1)))
    crossunder = (ema8 < ema20) & (ema8.shift(1).ge(ema20.shift(1)))

    # Direction filters
    long_cond = crossover & (ema20 > ema50)
    short_cond = crossunder & (ema20 < ema50)

    # Time filter based on hour
    times = pd.to_datetime(df['time'], unit='s')
    hour = times.dt.hour
    valid_time = ((hour >= 2) & (hour < 5)) | ((hour >= 10) & (hour < 12))

    long_entries = long_cond & valid_time
    short_entries = short_cond & valid_time

    # Ensure no NA values produce True
    long_entries = long_entries.fillna(False)
    short_entries = short_entries.fillna(False)

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if long_entries.iloc[i]:
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
                'raw_price_b': entry_price,
            })
            trade_num += 1
        elif short_entries.iloc[i]:
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
                'raw_price_b': entry_price,
            })
            trade_num += 1

    return entries