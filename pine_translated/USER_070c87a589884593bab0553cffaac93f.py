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
    results = []
    trade_num = 0

    # Calculate ATR (Wilder)
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr1 = tr.ewm(alpha=1/14, adjust=False).mean()

    # Calculate ATR for filter (atr211)
    atr2 = tr.ewm(alpha=1/20, adjust=False).mean() / 1.5

    # Volume filter
    vol_ma = volume.rolling(9).mean()
    volfilt = vol_ma * 1.5
    volfilt11 = volume.shift(1) > volfilt.shift(1)

    # ATR filter
    atrfilt11 = (low - high.shift(2) > atr2) | (low.shift(2) - high > atr2)

    # Trend filter (loc11)
    loc11 = close.rolling(54).mean()
    loc211 = loc11 > loc11.shift(1)
    locfiltb11 = loc211
    locfilts11 = ~loc211

    # Bullish FVG detection: low > high[2]
    bfvg11 = (low > high.shift(2)) & volfilt11 & atrfilt11 & locfiltb11

    # Bearish FVG detection: high < low[2]
    sfvg11 = (high < low.shift(2)) & volfilt11 & atrfilt11 & locfilts11

    # Skip bars where indicators are NaN (first 54 bars for loc11, first 20 for atr2)
    min_warmup = 54
    bfvg11.iloc[:min_warmup] = False
    sfvg11.iloc[:min_warmup] = False

    # Generate entries
    for i in range(len(df)):
        if bfvg11.iloc[i]:
            trade_num += 1
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })

        if sfvg11.iloc[i]:
            trade_num += 1
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })

    return results