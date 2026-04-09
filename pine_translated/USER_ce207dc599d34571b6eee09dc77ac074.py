import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    results = []
    trade_num = 0
    
    prd = 2
    max_array_size = 50
    
    zigzag = []
    
    for i in range(len(df)):
        high_price = df['high'].iloc[i]
        low_price = df['low'].iloc[i]
        close_price = df['close'].iloc[i]
        
        if i < prd:
            continue
        
        start_idx = i - prd + 1
        window_high = df['high'].iloc[start_idx:i+1].max()
        window_low = df['low'].iloc[start_idx:i+1].min()
        
        swing_high = high_price if high_price == window_high else np.nan
        swing_low = low_price if low_price == window_low else np.nan
        
        current_dir = 0
        if not np.isnan(swing_high) and np.isnan(swing_low):
            current_dir = 1
        elif not np.isnan(swing_low) and np.isnan(swing_high):
            current_dir = -1
        
        if current_dir == 0:
            continue
        
        if len(zigzag) == 0:
            zigzag.insert(0, swing_low if current_dir == -1 else swing_high)
            zigzag.insert(1, i)
        else:
            prev_dir = 1 if zigzag[0] > zigzag[2] else -1 if zigzag[0] < zigzag[2] else 0
            
            if current_dir != prev_dir:
                zigzag.insert(0, swing_low if current_dir == -1 else swing_high)
                zigzag.insert(1, i)
                if len(zigzag) > max_array_size:
                    zigzag.pop()
                    zigzag.pop()
            else:
                if len(zigzag) >= 2:
                    if current_dir == 1 and not np.isnan(swing_high) and swing_high > zigzag[0]:
                        zigzag[0] = swing_high
                        zigzag[1] = i
                    elif current_dir == -1 and not np.isnan(swing_low) and swing_low < zigzag[0]:
                        zigzag[0] = swing_low
                        zigzag[1] = i
        
        fib_50 = np.nan
        if len(zigzag) >= 6:
            fib_0 = zigzag[2]
            fib_1 = zigzag[4]
            if not (np.isnan(fib_0) or np.isnan(fib_1)):
                diff = fib_1 - fib_0
                fib_50 = fib_0 + diff * 0.5
        
        if not np.isnan(fib_50):
            if close_price < fib_50:
                trade_num += 1
                ts = int(df['time'].iloc[i])
                results.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': close_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close_price,
                    'raw_price_b': close_price
                })
            elif close_price > fib_50:
                trade_num += 1
                ts = int(df['time'].iloc[i])
                results.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': close_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close_price,
                    'raw_price_b': close_price
                })
    
    return results