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
    df = df.copy()
    df = df.reset_index(drop=True)
    
    inp1 = False  # Volume Filter
    inp2 = False  # ATR Filter
    inp3 = False  # Trend Filter
    
    # Volume Filter
    if inp1:
        volfilt = df['volume'].shift(1) > df['volume'].rolling(9).mean() * 1.5
    else:
        volfilt = pd.Series(True, index=df.index)
    
    # ATR Filter (Wilder)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    
    atr_raw = tr.ewm(alpha=1/20, adjust=False).mean()
    atr = atr_raw / 1.5
    
    if inp2:
        atrfilt = (df['low'] - df['high'].shift(2) > atr) | (df['low'].shift(2) - df['high'] > atr)
    else:
        atrfilt = pd.Series(True, index=df.index)
    
    # Trend Filter
    loc = df['close'].ewm(span=54, adjust=False).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2 if inp3 else pd.Series(True, index=df.index)
    locfilts = ~loc2 if inp3 else pd.Series(True, index=df.index)
    
    # Bullish and Bearish FVGs
    bull_fvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb
    bear_fvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts
    
    # State tracking variables
    lastFVG = 0  # 1 = Bullish, -1 = Bearish, 0 = None
    takenBullishTrade = False
    takenBearishTrade = False
    last_entry = None
    trade_num = 1
    
    entries = []
    
    for i in range(2, len(df)):
        bull_fvg_i = bull_fvg.iloc[i]
        bear_fvg_i = bear_fvg.iloc[i]
        lastFVG_prev = lastFVG
        takenBull_prev = takenBullishTrade
        takenBear_prev = takenBearishTrade
        last_entry_prev = last_entry
        
        # Detect if price entered bull FVG zone (bull box)
        # Bull box: top = low, bottom = high[2], entered when low < high[2]
        bull_box_entered = (df['low'].iloc[i] < df['high'].iloc[i-2]) if i >= 2 else False
        
        # Detect if price entered bear FVG zone (bear box)
        # Bear box: top = high[2], bottom = low, entered when high > low[2]
        bear_box_entered = (df['high'].iloc[i] > df['low'].iloc[i-2]) if i >= 2 else False
        
        # Update state
        if bull_box_entered:
            last_entry = "Entered Bullish FVG"
        elif bear_box_entered:
            last_entry = "Entered Bearish FVG"
        
        # Sharp turn long entry
        if bull_fvg_i and lastFVG_prev == -1 and last_entry_prev == "Entered Bullish FVG" and not takenBull_prev:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
            
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
            lastFVG = 1
            takenBullishTrade = True
            takenBearishTrade = False
        # Sharp turn short entry
        elif bear_fvg_i and lastFVG_prev == 1 and last_entry_prev == "Entered Bearish FVG" and not takenBear_prev:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
            
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
            lastFVG = -1
            takenBearishTrade = True
            takenBullishTrade = False
        else:
            # Update lastFVG if FVG detected but no trade taken
            if bull_fvg_i:
                lastFVG = 1
            elif bear_fvg_i:
                lastFVG = -1
    
    return entries