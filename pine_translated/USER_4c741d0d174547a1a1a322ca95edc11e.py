import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Parameters
    PP = 5  # Pivot Period
    
    # Initialize arrays for zigzag
    zigzag_type = []
    zigzag_value = []
    zigzag_index = []
    
    # Detect pivots
    high_pivot = df['high'].rolling(window=PP+1).max().shift(1) == df['high']
    low_pivot = df['low'].rolling(window=PP+1).min().shift(1) == df['low']
    
    # State tracking
    bull_bos = False
    bear_bos = False
    bull_choch = False
    bear_choch = False
    
    last_high_idx = 0
    last_low_idx = 0
    last_high_val = 0.0
    last_low_val = 0.0
    
    entries = []
    trade_num = 1
    
    for i in range(PP+1, len(df)):
        is_high_pivot = high_pivot.iloc[i]
        is_low_pivot = low_pivot.iloc[i]
        
        if is_high_pivot and is_low_pivot:
            high_val = df['high'].iloc[i]
            low_val = df['low'].iloc[i]
            high_idx = i
            low_idx = i
            
            if len(zigzag_type) == 0:
                if high_val > low_val:
                    zigzag_type.append('H')
                    zigzag_value.append(high_val)
                    zigzag_index.append(high_idx)
                else:
                    zigzag_type.append('L')
                    zigzag_value.append(low_val)
                    zigzag_index.append(low_idx)
            else:
                last_type = zigzag_type[-1]
                
                if last_type in ['L', 'LL']:
                    if low_val < zigzag_value[-1]:
                        zigzag_value[-1] = low_val
                        zigzag_index[-1] = low_idx
                        zigzag_type[-1] = 'LL' if len(zigzag_type) > 2 and zigzag_value[-3] < low_val else 'L'
                    else:
                        zigzag_type.append('H')
                        zigzag_value.append(high_val)
                        zigzag_index.append(high_idx)
                elif last_type in ['H', 'HH']:
                    if high_val > zigzag_value[-1]:
                        zigzag_value[-1] = high_val
                        zigzag_index[-1] = high_idx
                        zigzag_type[-1] = 'HH' if len(zigzag_type) > 2 and zigzag_value[-3] > high_val else 'H'
                    else:
                        zigzag_type.append('L')
                        zigzag_value.append(low_val)
                        zigzag_index.append(low_idx)
        
        # Update last high/low for structure detection
        if len(zigzag_index) >= 2:
            last_high_idx = zigzag_index[-1] if zigzag_type[-1] in ['H', 'HH'] else zigzag_index[-2]
            last_high_val = zigzag_value[-1] if zigzag_type[-1] in ['H', 'HH'] else zigzag_value[-2]
            last_low_idx = zigzag_index[-1] if zigzag_type[-1] in ['L', 'LL'] else zigzag_index[-2]
            last_low_val = zigzag_value[-1] if zigzag_type[-1] in ['L', 'LL'] else zigzag_value[-2]
        
        # Detect BoS and ChoCh
        if len(zigzag_type) >= 3:
            prev_type = zigzag_type[-2]
            curr_type = zigzag_type[-1]
            
            # Major BoS detection
            if curr_type == 'H' and prev_type == 'L':
                if zigzag_value[-1] > zigzag_value[-3]:
                    bull_bos = True
                    bear_bos = False
            elif curr_type == 'L' and prev_type == 'H':
                if zigzag_value[-1] < zigzag_value[-3]:
                    bear_bos = True
                    bull_bos = False
            
            # ChoCh detection (structure violation)
            if curr_type == 'H':
                if zigzag_value[-1] < zigzag_value[-2] and prev_type == 'L':
                    bull_choch = True
            elif curr_type == 'L':
                if zigzag_value[-1] > zigzag_value[-2] and prev_type == 'H':
                    bear_choch = True
        
        # Entry signals based on structure
        if bull_bos or bull_choch:
            ts = int(df['time'].iloc[i])
            entries.append({
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
            })
            trade_num += 1
            bull_bos = False
            bull_choch = False
        
        if bear_bos or bear_choch:
            ts = int(df['time'].iloc[i])
            entries.append({
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
            })
            trade_num += 1
            bear_bos = False
            bear_choch = False
    
    return entries