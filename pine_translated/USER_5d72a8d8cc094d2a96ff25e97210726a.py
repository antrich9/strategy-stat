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
    # Strategy parameters
    fastLength = 8
    mediumLength = 20
    slowLength = 50
    pipSize = 0.0002  # default for non‑JPY pairs

    # Compute EMAs
    fastEMA = df['close'].ewm(span=fastLength, adjust=False).mean()
    mediumEMA = df['close'].ewm(span=mediumLength, adjust=False).mean()
    slowEMA = df['close'].ewm(span=slowLength, adjust=False).mean()

    # Price‑action threshold
    upperThreshold = df['high'] - ((df['high'] - df['low']) * 0.31)

    # Bullish candle condition
    bullishCandle = (
        (df['close'] > upperThreshold) &
        (df['open'] > upperThreshold) &
        (df['low'] <= fastEMA)
    )

    # EMA alignment condition
    longEMAsAligned = (fastEMA > mediumEMA) & (mediumEMA > slowEMA)

    # Combined long entry condition
    longCondition = bullishCandle & longEMAsAligned

    # Date range filter (inclusive)
    start_ts = int(datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime(2023, 12, 31, 23, 59, 0, tzinfo=timezone.utc).timestamp())
    inDateRange = (df['time'] >= start_ts) & (df['time'] <= end_ts)

    # Final entry signals (ensure no NaN booleans)
    condition = (longCondition & inDateRange).fillna(False)

    entries = []
    trade_num = 1

    for i in df[condition].index:
        entry_ts = int(df.loc[i, 'time'])
        entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
        entry_price_guess = df.loc[i, 'high'] + pipSize
        raw_price_a = raw_price_b = entry_price_guess

        entries.append({
            'trade_num': trade_num,
            'direction': 'long',
            'entry_ts': entry_ts,
            'entry_time': entry_time,
            'entry_price_guess': entry_price_guess,
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': raw_price_a,
            'raw_price_b': raw_price_b,
        })
        trade_num += 1

    return entries