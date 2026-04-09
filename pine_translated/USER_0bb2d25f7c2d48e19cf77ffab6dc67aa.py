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
    
    # Input settings (matching Pine Script defaults)
    inp11 = False  # Volume Filter
    inp21 = False  # ATR Filter
    inp31 = False  # Trend Filter
    
    # Resample to 4H data using pandas
    df['time_dt'] = pd.to_datetime(df['time'], unit='s')
    df_4h = df.set_index('time_dt').resample('240T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    df_4h = df_4h.reset_index()
    df_4h['time'] = df_4h['time_dt'].astype(np.int64) // 10**9
    
    # Volume Filter (4H data)
    volfilt1 = df_4h['volume'].rolling(9).mean() * 1.5
    if inp11:
        volfilt1 = df_4h['volume'].shift(1) > volfilt1.shift(1)
    else:
        volfilt1 = pd.Series(True, index=df_4h.index)
    
    # ATR Filter (4H data) - Wilder ATR
    atr_length1 = 20
    high_4h = df_4h['high']
    low_4h = df_4h['low']
    close_4h = df_4h['close']
    
    tr = pd.concat([high_4h - low_4h, 
                    (high_4h - close_4h.shift(1)).abs(), 
                    (low_4h - close_4h.shift(1)).abs()], axis=1).max(axis=1)
    atr_4h = tr.ewm(alpha=1/atr_length1, adjust=False).mean() / 1.5
    
    if inp21:
        atrfilt1 = (low_4h - high_4h.shift(2) > atr_4h) | (low_4h.shift(2) - high_4h > atr_4h)
    else:
        atrfilt1 = pd.Series(True, index=df_4h.index)
    
    # Trend Filter (4H data)
    loc1 = close_4h.rolling(54).mean()
    loc21 = loc1 > loc1.shift(1)
    locfiltb1 = loc21 if inp31 else pd.Series(True, index=df_4h.index)
    locfilts1 = ~loc21 if inp31 else pd.Series(True, index=df_4h.index)
    
    # Identify Bullish and Bearish FVGs (4H data)
    bfvg1 = (low_4h > high_4h.shift(2)) & volfilt1 & atrfilt1 & locfiltb1
    sfvg1 = (high_4h < low_4h.shift(2)) & volfilt1 & atrfilt1 & locfilts1
    
    # Track last FVG state
    lastFVG = 0
    last_close_idx = 0
    
    # Detect new 4H candles
    is_new_4h1 = pd.Series(False, index=df_4h.index)
    is_new_4h1.iloc[1:] = True
    
    # Iterate through 4H bars
    for i in range(1, len(df_4h)):
        if not is_new_4h1.iloc[i]:
            continue
        
        current_bfvg = bfvg1.iloc[i]
        current_sfvg = sfvg1.iloc[i]
        
        # Sharp Turn detection
        if current_bfvg and lastFVG == -1:
            trade_num += 1
            entry_ts = int(df_4h['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price_guess = float(df_4h['close'].iloc[i])
            results.append({
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
            lastFVG = 1
        elif current_sfvg and lastFVG == 1:
            trade_num += 1
            entry_ts = int(df_4h['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price_guess = float(df_4h['close'].iloc[i])
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price_guess,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price_guess,
                'raw_price_b': entry_price_guess
            })
            lastFVG = -1
        elif current_bfvg:
            lastFVG = 1
        elif current_sfvg:
            lastFVG = -1
    
    return results