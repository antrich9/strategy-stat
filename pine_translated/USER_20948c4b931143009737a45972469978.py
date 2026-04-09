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
    trade_num = 1
    
    # Resample to 4H timeframe
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df_4h = df.set_index('datetime').resample('240min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()
    
    if len(df_4h) < 3:
        return results
    
    # Calculate 4H indicators
    # Volume Filter - using Wilder-like smoothing (sma on volume)
    volfilt_series = df_4h['volume'].rolling(9).mean() * 1.5
    
    # ATR Filter - using Wilder ATR manually
    high_4h = df_4h['high']
    low_4h = df_4h['low']
    prev_high_2 = high_4h.shift(2)
    prev_low_2 = low_4h.shift(2)
    
    atr_length = 20
    tr = pd.concat([high_4h - low_4h, abs(high_4h - prev_low_2), abs(prev_high_2 - low_4h)], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/atr_length, adjust=False).mean() / 1.5
    
    # Trend Filter - using Wilder SMA (54 period)
    loc1 = df_4h['close'].rolling(54).mean()
    loc21 = loc1 > loc1.shift(1)
    
    # Fair Value Gap conditions (4H data)
    bfvg1 = (low_4h > prev_high_2) & (df_4h['volume'].shift(1) > volfilt_series) & \
            ((low_4h - prev_high_2 > atr) | (prev_low_2 - high_4h > atr)) & loc21
    
    sfvg1 = (high_4h < prev_low_2) & (df_4h['volume'].shift(1) > volfilt_series) & \
            ((low_4h - prev_high_2 > atr) | (prev_low_2 - high_4h > atr)) & ~loc21
    
    # Also need 15min data filters for additional conditions
    volfilt = df['volume'].shift(1) > df['volume'].rolling(9).mean() * 1.5
    tr_15 = pd.concat([df['high'] - df['low'], abs(df['high'] - df['low'].shift(2)), 
                      abs(df['high'].shift(2) - df['low'])], axis=1).max(axis=1)
    atr_15 = tr_15.ewm(alpha=1.0/20, adjust=False).mean() / 1.5
    atrfilt = (df['low'] - df['high'].shift(2) > atr_15) | (df['low'].shift(2) - df['high'] > atr_15)
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    # 15min FVG
    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts
    
    # Track last FVG type
    lastFVG = 0
    trade_active = False
    entry_idx = 0
    
    for i in range(16, len(df)):
        if pd.isna(bfvg.iloc[i]) or pd.isna(sfvg.iloc[i]):
            continue
        
        # Detect new 4H candle
        current_ts = df['time'].iloc[i]
        current_dt = datetime.fromtimestamp(current_ts, tz=timezone.utc)
        prev_ts = df['time'].iloc[i-1] if i > 0 else current_ts
        prev_dt = datetime.fromtimestamp(prev_ts, tz=timezone.utc)
        
        is_new_4h = (current_dt.hour % 4 == 0 and current_dt.minute == 0) and \
                    (prev_dt.hour % 4 != 0 or prev_dt.minute != 0)
        
        if i > 0:
            prev_hour = datetime.fromtimestamp(df['time'].iloc[i-1], tz=timezone.utc).hour
            is_new_4h = (current_dt.hour % 4 == 0 and current_dt.minute == 0) and \
                        (prev_hour % 4 != 0 or prev_dt.minute != 0)
        
        # Get corresponding 4H index
        fvg_bull = bfvg.iloc[i] if not pd.isna(bfvg.iloc[i]) else False
        fvg_bear = sfvg.iloc[i] if not pd.isna(sfvg.iloc[i]) else False
        
        if is_new_4h and not trade_active:
            # Check for sharp turn - long entry
            if fvg_bull and lastFVG == -1:
                entry_price = df['close'].iloc[i]
                entry_ts = int(df['time'].iloc[i])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                
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
                trade_num += 1
                trade_active = True
                entry_idx = i
                lastFVG = 1
            # Check for sharp turn - short entry
            elif fvg_bear and lastFVG == 1:
                entry_price = df['close'].iloc[i]
                entry_ts = int(df['time'].iloc[i])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                
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
                trade_num += 1
                trade_active = True
                entry_idx = i
                lastFVG = -1
            # Update lastFVG without entry
            elif fvg_bull:
                lastFVG = 1
            elif fvg_bear:
                lastFVG = -1
        
    return results