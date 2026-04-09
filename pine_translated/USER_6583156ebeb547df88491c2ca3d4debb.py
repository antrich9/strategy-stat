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
    PP = 6
    
    # Initialize ZigZag arrays
    zigzag_type = []
    zigzag_value = []
    zigzag_index = []
    
    # Variables for Major levels
    major_high = np.nan
    major_low = np.nan
    major_high_idx = -1
    major_low_idx = -1
    
    # BoS and ChoCh flags
    bullish_major_bos = False
    bearish_major_bos = False
    bullish_major_choch = False
    bearish_major_choch = False
    
    entries = []
    trade_num = 1
    
    n = len(df)
    
    # Pre-compute pivots using rolling windows
    high_pivots = np.zeros(n)
    low_pivots = np.zeros(n)
    
    for i in range(PP, n - PP):
        is_high = True
        is_low = True
        for j in range(1, PP + 1):
            if df['high'].iloc[i] <= df['high'].iloc[i - j]:
                is_high = False
            if df['high'].iloc[i] <= df['high'].iloc[i + j]:
                is_high = False
            if df['low'].iloc[i] >= df['low'].iloc[i - j]:
                is_low = False
            if df['low'].iloc[i] >= df['low'].iloc[i + j]:
                is_low = False
        if is_high:
            high_pivots[i] = df['high'].iloc[i]
        if is_low:
            low_pivots[i] = df['low'].iloc[i]
    
    # Iterate through bars
    for i in range(PP, n):
        current_high = high_pivots[i]
        current_low = low_pivots[i]
        
        # Build zigzag arrays when pivot detected
        if current_high > 0 or current_low > 0:
            if len(zigzag_type) == 0:
                if current_high > 0:
                    zigzag_type.append('H')
                    zigzag_value.append(current_high)
                    zigzag_index.append(i)
                elif current_low > 0:
                    zigzag_type.append('L')
                    zigzag_value.append(current_low)
                    zigzag_index.append(i)
            else:
                last_type = zigzag_type[-1]
                last_value = zigzag_value[-1]
                
                if current_high > 0:
                    if last_type == 'H':
                        if current_high > last_value:
                            zigzag_value[-1] = current_high
                            zigzag_index[-1] = i
                        else:
                            zigzag_type.append('H')
                            zigzag_value.append(current_high)
                            zigzag_index.append(i)
                    else:
                        zigzag_type.append('H')
                        zigzag_value.append(current_high)
                        zigzag_index.append(i)
                
                elif current_low > 0:
                    if last_type == 'L':
                        if current_low < last_value:
                            zigzag_value[-1] = current_low
                            zigzag_index[-1] = i
                        else:
                            new_type = 'LH' if (len(zigzag_type) > 1 and zigzag_value[-2] < current_low) else 'LL'
                            zigzag_type[-1] = new_type
                            zigzag_value[-1] = current_low
                            zigzag_index[-1] = i
                    else:
                        zigzag_type.append('L')
                        zigzag_value.append(current_low)
                        zigzag_index.append(i)
        
        # Update Major levels when we have enough pivots
        if len(zigzag_index) >= 4:
            major_high_idx = zigzag_index[-2]
            major_low_idx = zigzag_index[-3]
            major_high = zigzag_value[-2]
            major_low = zigzag_value[-3]
        
        # Detect BoS and ChoCh
        if len(zigzag_index) >= 5:
            prev_idx = zigzag_index[-3]
            prev_value = zigzag_value[-3]
            
            if df['close'].iloc[i] > prev_value and df['high'].iloc[i] > df['high'].iloc[prev_idx]:
                bullish_major_bos = True
            
            if df['close'].iloc[i] < prev_value and df['low'].iloc[i] < df['low'].iloc[prev_idx]:
                bearish_major_bos = True
        
        # Check for ChoCh (Change of Character)
        if len(zigzag_type) >= 4:
            second_last_type = zigzag_type[-3]
            last_type = zigzag_type[-2]
            
            if second_last_type == 'H' and last_type == 'L':
                bullish_major_choch = True
                bearish_major_choch = False
            elif second_last_type == 'L' and last_type == 'H':
                bearish_major_choch = True
                bullish_major_choch = False
        
        # Entry conditions (simplified)
        if i >= PP:
            is_bullish_entry = False
            is_bearish_entry = False
            
            # Bullish: Break of structure after ChoCh
            if len(zigzag_index) >= 5:
                prev_idx = zigzag_index[-3]
                prev_value = zigzag_value[-3]
                if bullish_major_bos and bullish_major_choch:
                    if df['close'].iloc[i] > prev_value:
                        is_bullish_entry = True
            
            # Bearish: Break of structure after ChoCh
            if len(zigzag_index) >= 5:
                prev_idx = zigzag_index[-3]
                prev_value = zigzag_value[-3]
                if bearish_major_bos and bearish_major_choch:
                    if df['close'].iloc[i] < prev_value:
                        is_bearish_entry = True
            
            # Additional entry: Strong momentum break
            if not is_bullish_entry and len(zigzag_index) >= 4:
                last_high_idx = zigzag_index[-2]
                if df['close'].iloc[i] > zigzag_value[-2] and df['volume'].iloc[i] > df['volume'].iloc[i-1]:
                    is_bullish_entry = True
            
            if not is_bearish_entry and len(zigzag_index) >= 4:
                last_low_idx = zigzag_index[-2]
                if df['close'].iloc[i] < zigzag_value[-2] and df['volume'].iloc[i] > df['volume'].iloc[i-1]:
                    is_bearish_entry = True
            
            # Execute entries
            if is_bullish_entry:
                ts = int(df['time'].iloc[i])
                entry = {
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(df['close'].iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(df['close'].iloc[i]),
                    'raw_price_b': float(df['close'].iloc[i])
                }
                entries.append(entry)
                trade_num += 1
            
            if is_bearish_entry:
                ts = int(df['time'].iloc[i])
                entry = {
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(df['close'].iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(df['close'].iloc[i]),
                    'raw_price_b': float(df['close'].iloc[i])
                }
                entries.append(entry)
                trade_num += 1
    
    return entries