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
    # Ensure chronological order
    df = df.sort_values('time').reset_index(drop=True)

    close = df['close']

    # EMA lengths from the strategy
    fastLength = 8
    mediumLength = 20
    slowLength = 50

    # Exponential moving averages
    fastEMA = close.ewm(span=fastLength, adjust=False).mean()
    mediumEMA = close.ewm(span=mediumLength, adjust=False).mean()
    slowEMA = close.ewm(span=slowLength, adjust=False).mean()

    # Trend confirmation filters
    isBullishTrend = ((close > slowEMA) & (fastEMA > mediumEMA) & (mediumEMA > slowEMA)).fillna(False)
    isBearishTrend = ((close < slowEMA) & (fastEMA < mediumEMA) & (mediumEMA < slowEMA)).fillna(False)

    # Crossover / crossunder of price vs fast EMA
    crossover = ((close > fastEMA) & (close.shift(1) <= fastEMA.shift(1))).fillna(False)
    crossunder = ((close < fastEMA) & (close.shift(1) >= fastEMA.shift(1))).fillna(False)

    # Time window filter (London 07‑10 and NY 14‑17)
    times = pd.to_datetime(df['time'], unit='s', utc=True)
    hour = times.dt.hour
    in_trading_window = (((hour >= 7) & (hour <= 10)) | ((hour >= 14) & (hour <= 17))).fillna(False)

    # Entry conditions
    longCondition = (crossover & isBullishTrend & in_trading_window).fillna(False)
    shortCondition = (crossunder & isBearishTrend & in_trading_window).fillna(False)

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if longCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif shortCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

    return entries