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
    df = df.copy()

    # EMAs
    fastEMA = df['close'].ewm(span=9, adjust=False).mean()
    slowEMA = df['close'].ewm(span=18, adjust=False).mean()

    # Trend conditions
    isWeakBullishTrend = (df['close'] > slowEMA) & (fastEMA > slowEMA)
    isWeakBearishTrend = (df['close'] < slowEMA) & (fastEMA < slowEMA)

    # Crossover / crossunder
    crossover = (df['close'] > fastEMA) & (df['close'].shift(1) <= fastEMA.shift(1))
    crossunder = (df['close'] < fastEMA) & (df['close'].shift(1) >= fastEMA.shift(1))

    # Entry condition series
    long_condition = (crossover & isWeakBullishTrend).fillna(False).astype(bool)
    short_condition = (crossunder & isWeakBearishTrend).fillna(False).astype(bool)

    entries = []
    trade_num = 1

    for i, row in df.iterrows():
        if pd.isna(fastEMA.iloc[i]) or pd.isna(slowEMA.iloc[i]):
            continue

        if long_condition.iloc[i]:
            entry_ts = int(row['time'])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(row['close']),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(row['close']),
                'raw_price_b': float(row['close'])
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            entry_ts = int(row['time'])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(row['close']),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(row['close']),
                'raw_price_b': float(row['close'])
            })
            trade_num += 1

    return entries