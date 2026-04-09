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
    entries = []
    trade_num = 0
    
    # Volume filter
    volfilt = df['volume'] > df['volume'].shift(1).rolling(9).mean() * 1.5
    
    # ATR filter (Wilder)
    tr1 = df['high'] - df['low']
    tr2 = np.abs(df['high'] - df['close'].shift(1))
    tr3 = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/20, adjust=False).mean()
    atr_filt_val = atr / 1.5
    atrfilt = (df['low'] - df['high'].shift(2) > atr_filt_val) | (df['low'].shift(2) - df['high'] > atr_filt_val)
    
    # Trend filter
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    # FVG conditions
    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts
    
    # State variables
    lastFVG = 0
    takenBullishTrade = False
    takenBearishTrade = False
    last_entry = None
    
    # Track FVG boxes and entry states
    bull_box_top = np.full(len(df), np.nan)
    bull_box_bottom = np.full(len(df), np.nan)
    bear_box_top = np.full(len(df), np.nan)
    bear_box_bottom = np.full(len(df), np.nan)
    bull_entered = np.full(len(df), False)
    bear_entered = np.full(len(df), False)
    
    for i in range(1, len(df)):
        if i < 2:
            continue
            
        # Detect FVG at bar i-2 (confirmed bar)
        x = 0
        if df['low'].iloc[i-2] >= df['high'].iloc[i]:
            x = -1
        elif df['high'].iloc[i-2] <= df['low'].iloc[i]:
            x = 1
            
        if x > 0:
            bull_box_top[i-2] = df['low'].iloc[i]
            bull_box_bottom[i-2] = df['high'].iloc[i-2]
        elif x < 0:
            bear_box_top[i-2] = df['high'].iloc[i]
            bear_box_bottom[i-2] = df['low'].iloc[i-2]
    
    for i in range(14, len(df)):
        if pd.isna(bfvg.iloc[i]) or pd.isna(sfvg.iloc[i]):
            continue
            
        # Check if price entered bullish FVG
        for j in range(i):
            if not pd.isna(bull_box_top[j]) and not bull_entered[j]:
                if df['low'].iloc[i] < bull_box_top[j]:
                    bull_entered[j] = True
                    lastFVG = 1
                    last_entry = "Entered Bullish FVG"
                    break
        
        # Check if price entered bearish FVG
        for j in range(i):
            if not pd.isna(bear_box_bottom[j]) and not bear_entered[j]:
                if df['high'].iloc[i] > bear_box_bottom[j]:
                    bear_entered[j] = True
                    lastFVG = -1
                    last_entry = "Entered Bearish FVG"
                    break
        
        # Entry conditions
        if bfvg.iloc[i] and lastFVG == -1 and last_entry == "Entered Bullish FVG" and not takenBullishTrade:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            takenBullishTrade = True
            takenBearishTrade = False
        elif sfvg.iloc[i] and lastFVG == 1 and last_entry == "Entered Bearish FVG" and not takenBearishTrade:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            takenBearishTrade = True
            takenBullishTrade = False
    
    return entries