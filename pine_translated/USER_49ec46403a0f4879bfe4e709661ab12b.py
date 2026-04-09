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
    fastLength = 50
    slowLength = 200

    # Simple moving averages
    fastMA = df['close'].rolling(window=fastLength, min_periods=fastLength).mean()
    slowMA = df['close'].rolling(window=slowLength, min_periods=slowLength).mean()

    # Crossover signal: fastMA crosses above slowMA
    buySignal = (fastMA > slowMA) & (fastMA.shift(1) <= slowMA.shift(1))

    entries = []
    trade_num = 1

    for i in df.index:
        if pd.isna(buySignal.iloc[i]) or not buySignal.iloc[i]:
            continue
        entry_ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
        entry_price_guess = float(df['close'].iloc[i])

        entries.append({
            'trade_num': trade_num,
            'direction': 'long',
            'entry_ts': entry_ts,
            'entry_time': entry_time,
            'entry_price_guess': entry_price_guess,
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': entry_price_guess,
            'raw_price_b': entry_price_guess
        })
        trade_num += 1

    return entries