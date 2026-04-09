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
    open_series = df['open']
    high_series = df['high']
    low_series = df['low']
    close_series = df['close']
    volume_series = df['volume']

    volfilt = True  # inp1 defaults to false
    atrfilt = True  # inp2 defaults to false

    loc = close_series.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = True  # inp3 defaults to false
    locfilts = True  # inp3 defaults to false

    bfvg = (low_series > high_series.shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (high_series < low_series.shift(2)) & volfilt & atrfilt & locfilts

    entries = []
    trade_num = 1
    lastFVG = 0

    for i in range(2, len(df)):
        if bfvg.iloc[i]:
            if lastFVG == -1:
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
                lastFVG = 1
            else:
                lastFVG = 1
        elif sfvg.iloc[i]:
            if lastFVG == 1:
                ts = int(df['time'].iloc[i])
                entry_price = float(df['close'].iloc[i])
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
                lastFVG = -1
            else:
                lastFVG = -1

    return entries