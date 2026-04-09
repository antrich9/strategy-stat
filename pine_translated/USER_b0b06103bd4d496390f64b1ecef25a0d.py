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
    pp = 6
    
    # Calculate pivots
    high_pivot = (df['high'] == df['high'].rolling(pp*2+1, center=True).max()) & \
                 (df['high'].shift(pp) == df['high'].rolling(pp*2+1, center=True).max())
    low_pivot = (df['low'] == df['low'].rolling(pp*2+1, center=True).min()) & \
                (df['low'].shift(pp) == df['low'].rolling(pp*2+1, center=True).min())
    
    high_pivot = high_pivot.fillna(False)
    low_pivot = low_pivot.fillna(False)
    
    # Initialize arrays for zigzag
    zigzag_type = [None] * len(df)
    zigzag_value = [np.nan] * len(df)
    zigzag_index = [0] * len(df)
    
    # Build zigzag
    last_type = None
    for i in range(len(df)):
        if high_pivot.iloc[i] or low_pivot.iloc[i]:
            if last_type is None:
                if high_pivot.iloc[i]:
                    zigzag_type[i] = 'H'
                    zigzag_value[i] = df['high'].iloc[i]
                    zigzag_index[i] = i
                    last_type = 'H'
                elif low_pivot.iloc[i]:
                    zigzag_type[i] = 'L'
                    zigzag_value[i] = df['low'].iloc[i]
                    zigzag_index[i] = i
                    last_type = 'L'
            else:
                if high_pivot.iloc[i] and last_type == 'L':
                    zigzag_type[i] = 'H'
                    zigzag_value[i] = df['high'].iloc[i]
                    zigzag_index[i] = i
                    last_type = 'H'
                elif low_pivot.iloc[i] and last_type == 'H':
                    zigzag_type[i] = 'L'
                    zigzag_value[i] = df['low'].iloc[i]
                    zigzag_index[i] = i
                    last_type = 'L'
    
    zigzag_type_series = pd.Series(zigzag_type)
    zigzag_value_series = pd.Series(zigzag_value)
    zigzag_index_series = pd.Series(zigzag_index)
    
    # Detect structure (simplified)
    bull_bos = np.zeros(len(df), dtype=bool)
    bear_bos = np.zeros(len(df), dtype=bool)
    bull_choch = np.zeros(len(df), dtype=bool)
    bear_choch = np.zeros(len(df), dtype=bool)
    
    # Wilder ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1.0/55, adjust=False).mean()
    
    for i in range(pp * 2 + 10, len(df)):
        # Find recent pivots
        recent_h = []
        recent_l = []
        for j in range(i - 1, max(0, i - 50), -1):
            if zigzag_type_series.iloc[j] == 'H':
                recent_h.append((j, zigzag_value_series.iloc[j]))
            elif zigzag_type_series.iloc[j] == 'L':
                recent_l.append((j, zigzag_value_series.iloc[j]))
            if len(recent_h) >= 3 and len(recent_l) >= 3:
                break
        
        if len(recent_h) >= 2 and len(recent_l) >= 2:
            h1, v1 = recent_h[0]
            h2, v2 = recent_h[1]
            l1, lv1 = recent_l[0]
            l2, lv2 = recent_l[1]
            
            # Bullish BoS
            if v2 < v1 and df['close'].iloc[i] > v1:
                bull_bos[i] = True
            
            # Bearish BoS
            if lv2 > lv1 and df['close'].iloc[i] < lv1:
                bear_bos[i] = True
            
            # Bullish ChoCh
            if len(recent_l) >= 3 and len(recent_h) >= 2:
                h1, v1 = recent_h[0]
                h2, v2 = recent_h[1]
                l1, lv1 = recent_l[0]
                l2, lv2 = recent_l[1]
                l3, lv3 = recent_l[2]
                if lv3 > lv2 and lv1 < lv2:
                    bull_choch[i] = True
            
            # Bearish ChoCh
            if len(recent_h) >= 3 and len(recent_l) >= 2:
                h1, v1 = recent_h[0]
                h2, v2 = recent_h[1]
                h3, v3 = recent_h[2]
                l1, lv1 = recent_l[0]
                l2, lv2 = recent_l[1]
                if v3 > v2 and v1 < v2:
                    bear_choch[i] = True
    
    # Order blocks (simplified)
    bull_ob = np.zeros(len(df), dtype=bool)
    bear_ob = np.zeros(len(df), dtype=bool)
    
    for i in range(pp + 5, len(df)):
        # Bearish OB: up bar followed by down move
        if df['close'].iloc[i] > df['open'].iloc[i]:
            move_down = df['high'].iloc[i-pp:i].max() > df['close'].iloc[i]
            if move_down:
                bear_ob[i-pp:i+1] = True
        
        # Bullish OB: down bar followed by up move
        if df['close'].iloc[i] < df['open'].iloc[i]:
            move_up = df['low'].iloc[i-pp:i].min() < df['close'].iloc[i]
            if move_up:
                bull_ob[i-pp:i+1] = True
    
    bull_bos = pd.Series(bull_bos)
    bear_bos = pd.Series(bear_bos)
    bull_choch = pd.Series(bull_choch)
    bear_choch = pd.Series(bear_choch)
    bull_ob = pd.Series(bull_ob)
    bear_ob = pd.Series(bear_ob)
    
    # Entry signals
    long_entry = (bull_bos | bull_choch) & bull_ob
    long_entry = long_entry.fillna(False)
    
    short_entry = (bear_bos | bear_choch) & bear_ob
    short_entry = short_entry.fillna(False)
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if i < 1:
            continue
        
        direction = None
        if long_entry.iloc[i]:
            direction = 'long'
        elif short_entry.iloc[i]:
            direction = 'short'
        
        if direction:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
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