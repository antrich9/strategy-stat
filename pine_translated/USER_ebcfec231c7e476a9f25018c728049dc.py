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
    # Default parameters (match the strategy's default inputs)
    swing_length = 2
    inp1 = False  # Volume Filter disabled
    inp2 = False  # ATR Filter disabled
    inp3 = False  # Trend Filter disabled

    # Compute shifted series for FVG detection
    high_shift2 = df['high'].shift(2)
    low_shift2 = df['low'].shift(2)

    # Volume filter (only if enabled)
    if inp1:
        vol_sma = df['volume'].rolling(9).mean()
        volfilt = df['volume'].shift(1) > vol_sma * 1.5
    else:
        volfilt = pd.Series(True, index=df.index)

    # ATR filter (only if enabled)
    if inp2:
        # True Range using high, low, close
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/20, adjust=False).mean() / 1.5
        atrfilt = ((df['low'] - high_shift2) > atr) | ((low_shift2 - df['high']) > atr)
    else:
        atrfilt = pd.Series(True, index=df.index)

    # Trend filter (only if enabled)
    if inp3:
        sma54 = df['close'].rolling(54).mean()
        loc2 = sma54 > sma54.shift(1)
        locfiltb = loc2
        locfilts = ~loc2
    else:
        locfiltb = pd.Series(True, index=df.index)
        locfilts = pd.Series(True, index=df.index)

    # Fair Value Gap conditions
    bfvg = (df['low'] > high_shift2) & volfilt & atrfilt & locfiltb
    sfvg = (df['high'] < low_shift2) & volfilt & atrfilt & locfilts

    # Initialize state variables
    lastFVG = 0
    entries = []
    trade_num = 1

    # Iterate over bars (start at index 2 to have valid shifted values)
    for i in range(2, len(df)):
        is_bfvg = bfvg.iloc[i]
        is_sfvg = sfvg.iloc[i]

        prev_lastFVG = lastFVG

        # Long entry: bullish FVG after a bearish FVG
        if is_bfvg and prev_lastFVG == -1:
            entry_price = df['close'].iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
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
        # Short entry: bearish FVG after a bullish FVG
        elif is_sfvg and prev_lastFVG == 1:
            entry_price = df['close'].iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
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

        # Update lastFVG after entry check
        if is_bfvg:
            lastFVG = 1
        elif is_sfvg:
            lastFVG = -1

    return entries