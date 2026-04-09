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

    # Parameters
    fastLength = 8
    mediumLength = 20
    slowLength = 50

    # pipSize - default to non-JPY (0.0002) since no currency info in df
    pipSize = 0.0002

    # Date range: Jan 1, 2020 to Dec 31, 2023
    start_ts = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime(2023, 12, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp())

    # Calculate EMAs
    fastEMA = df['close'].ewm(span=fastLength, adjust=False).mean()
    mediumEMA = df['close'].ewm(span=mediumLength, adjust=False).mean()
    slowEMA = df['close'].ewm(span=slowLength, adjust=False).mean()

    # Price action pattern thresholds
    upperThreshold = df['high'] - ((df['high'] - df['low']) * 0.31)
    lowerThreshold = df['low'] + ((df['high'] - df['low']) * 0.31)

    # Bullish and bearish candle conditions
    bullishCandle = (df['close'] > upperThreshold) & (df['open'] > upperThreshold) & (df['low'] <= fastEMA)
    bearishCandle = (df['close'] < lowerThreshold) & (df['open'] < lowerThreshold) & (df['high'] >= fastEMA)

    # EMA alignment conditions
    longEMAsAligned = (fastEMA > mediumEMA) & (mediumEMA > slowEMA)
    shortEMAsAligned = (fastEMA < mediumEMA) & (mediumEMA < slowEMA)

    # Date range filter
    inDateRange = (df['time'] >= start_ts) & (df['time'] <= end_ts)

    # Combined entry conditions
    longCondition = bullishCandle & longEMAsAligned & inDateRange
    shortCondition = bearishCandle & shortEMAsAligned & inDateRange

    # Calculate entry prices (stop order prices)
    longEntryPrice = df['high'] + pipSize
    shortEntryPrice = df['low'] - pipSize

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if pd.isna(fastEMA.iloc[i]) or pd.isna(mediumEMA.iloc[i]) or pd.isna(slowEMA.iloc[i]):
            continue

        if longCondition.iloc[i]:
            entry_price = longEntryPrice.iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

        if shortCondition.iloc[i]:
            entry_price = shortEntryPrice.iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries