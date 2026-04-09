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
    df = df.sort_values('time').reset_index(drop=True)

    # Filter switches (default to False as per script)
    inp1 = False
    inp2 = False
    inp3 = False

    # Volume filter
    if inp1:
        volfilt = df['volume'].shift(1) > df['volume'].rolling(9).mean() * 1.5
    else:
        volfilt = pd.Series(True, index=df.index)

    # ATR (Wilder)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean()

    # ATR filter
    if inp2:
        atrfilt = ((df['low'] - df['high'].shift(2) > atr) | (df['low'].shift(2) - df['high'] > atr))
    else:
        atrfilt = pd.Series(True, index=df.index)

    # Trend filter
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)

    if inp3:
        locfiltb = loc2
        locfilts = ~loc2
    else:
        locfiltb = pd.Series(True, index=df.index)
        locfilts = pd.Series(True, index=df.index)

    # Bullish FVG (long) and Bearish FVG (short) entry conditions
    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if bfvg.iloc[i]:
            entry_price = df['close'].iloc[i]
            ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        if sfvg.iloc[i]:
            entry_price = df['close'].iloc[i]
            ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
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