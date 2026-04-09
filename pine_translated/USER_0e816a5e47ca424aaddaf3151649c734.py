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
    # Convert timestamps to New York time to extract hour and day-of-week
    ts = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert('America/New_York')
    hour = ts.dt.hour
    dayofweek = ts.dt.dayofweek  # Monday=0, ..., Sunday=6

    # Kill zone parameters (NY time)
    killZoneStart = 15
    killZoneEnd = 17

    # Determine if we are within the kill zone: Monday-Friday between start and end-1 hours
    inKillZone = (hour >= killZoneStart) & (hour < killZoneEnd) & (dayofweek >= 0) & (dayofweek <= 4)

    # Placeholder liquidity sweep (always False in this skeleton)
    sweepOccurred = pd.Series(False, index=df.index)

    # Placeholder entry condition (always False in this skeleton)
    entryCondition = pd.Series(False, index=df.index)

    # Long entry: sweep + entry condition + close below open + inside kill zone
    longEntry = sweepOccurred & entryCondition & (df['close'] < df['open']) & inKillZone

    # Short entry: sweep + entry condition + close above open + inside kill zone
    shortEntry = sweepOccurred & entryCondition & (df['close'] > df['open']) & inKillZone

    # Build list of entries
    entries = []
    trade_num = 1

    for i in df.index:
        if longEntry.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif shortEntry.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
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