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
    # Define date range timestamps (seconds)
    start_ts = int(datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime(2023, 12, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp())

    # Determine pip size based on typical price level
    pipSize = 0.02 if df['close'].mean() > 100 else 0.0002

    # EMAs (Wilder's EMA)
    fastEMA = df['close'].ewm(span=8, adjust=False).mean()
    mediumEMA = df['close'].ewm(span=20, adjust=False).mean()
    slowEMA = df['close'].ewm(span=50, adjust=False).mean()

    # Price action thresholds
    upperThreshold = df['high'] - ((df['high'] - df['low']) * 0.31)
    lowerThreshold = df['low'] + ((df['high'] - df['low']) * 0.31)

    # Bullish / Bearish candle conditions
    bullishCandle = (df['close'] > upperThreshold) & (df['open'] > upperThreshold) & (df['low'] <= fastEMA)
    bearishCandle = (df['close'] < lowerThreshold) & (df['open'] < lowerThreshold) & (df['high'] >= fastEMA)

    # EMA alignment
    longEMAsAligned = (fastEMA > mediumEMA) & (mediumEMA > slowEMA)
    shortEMAsAligned = (fastEMA < mediumEMA) & (mediumEMA < slowEMA)

    # Date range filter
    inDateRange = (df['time'] >= start_ts) & (df['time'] <= end_ts)

    # Ensure indicators are not NaN
    valid = fastEMA.notna() & mediumEMA.notna() & slowEMA.notna()

    # Combined entry conditions
    long_cond = (bullishCandle & longEMAsAligned & inDateRange & valid).fillna(False)
    short_cond = (bearishCandle & shortEMAsAligned & inDateRange & valid).fillna(False)

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if long_cond.iloc[i]:
            entry_price = df['high'].iloc[i] + pipSize
            ts = df['time'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        if short_cond.iloc[i]:
            entry_price = df['low'].iloc[i] - pipSize
            ts = df['time'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries