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
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/200, adjust=False).mean()

    filterWidth = 0.0

    bull = (
        (low.shift(3) > high.shift(1)) &
        (close.shift(2) < low.shift(3)) &
        (close > low.shift(3)) &
        ((low.shift(3) - high.shift(1)) > atr * filterWidth)
    )

    bear = (
        (low.shift(1) > high.shift(3)) &
        (close.shift(2) > high.shift(3)) &
        (close < high.shift(3)) &
        ((low.shift(1) - high.shift(3)) > atr * filterWidth)
    )

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if bull.iloc[i] and not pd.isna(atr.iloc[i]):
            entry_price = close.iloc[i]
            ts = df['time'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

        if bear.iloc[i] and not pd.isna(atr.iloc[i]):
            entry_price = close.iloc[i]
            ts = df['time'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(ts),
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