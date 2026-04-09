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
    atrPeriod = 14
    atrMultiplier = 3.0

    fastEMA = df['close'].ewm(span=fastLength, adjust=False).mean()
    slowEMA = df['close'].ewm(span=slowLength, adjust=False).mean()

    crossOver = (fastEMA > slowEMA) & (fastEMA.shift(1) <= slowEMA.shift(1))
    crossUnder = (fastEMA < slowEMA) & (fastEMA.shift(1) >= slowEMA.shift(1))

    tr = np.maximum(df['high'] - df['low'], np.abs(df['high'] - df['close'].shift(1)))
    tr = np.maximum(tr, np.abs(df['low'] - df['close'].shift(1)))
    atr = tr.ewm(alpha=1/atrPeriod, adjust=False).mean()

    entries = []
    trade_num = 1
    in_position = False

    for i in range(1, len(df)):
        if crossOver.iloc[i] and not in_position:
            ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
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
            in_position = True
        elif crossUnder.iloc[i] and in_position:
            in_position = False

    return entries