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
    # Initialize filters (all disabled by default as per input.bool(false, ...))
    inp1 = False  # Volume Filter
    inp2 = False  # ATR Filter
    inp3 = False  # Trend Filter

    # Calculate indicators
    # Volume Filter
    if inp1:
        volfilt = df['volume'].shift(1) > df['volume'].rolling(9).mean() * 1.5
    else:
        volfilt = pd.Series(True, index=df.index)

    # ATR Filter (Wilder ATR)
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_period = 20
    atr = tr.ewm(alpha=1/atr_period, adjust=False).mean()
    atr2 = atr / 1.5
    
    if inp2:
        atrfilt = (low - high.shift(2) > atr2) | (low.shift(2) - high > atr2)
    else:
        atrfilt = pd.Series(True, index=df.index)

    # Trend Filter (SMA 54)
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    
    if inp3:
        locfiltb = loc2
        locfilts = ~loc2
    else:
        locfiltb = pd.Series(True, index=df.index)
        locfilts = pd.Series(True, index=df.index)

    # Bullish FVG: bar_index >= 2 and low > high[2] and volfilt and atrfilt and locfiltb
    # Bearish FVG: bar_index >= 2 and high < low[2] and volfilt and atrfilt and locfilts
    bfvg = (df.index >= 2) & (low > high.shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (df.index >= 2) & (high < low.shift(2)) & volfilt & atrfilt & locfilts

    # Generate entries
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if pd.isna(bfvg.iloc[i]) or pd.isna(sfvg.iloc[i]):
            continue

        if bfvg.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            
            entries.append({
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
            trade_num += 1

        if sfvg.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            
            entries.append({
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
            trade_num += 1

    return entries