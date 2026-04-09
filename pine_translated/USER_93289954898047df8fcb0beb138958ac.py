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
    
    pp = 5
    
    n = len(df)
    if n < pp * 2 + 1:
        return []
    
    high = df['high'].values.copy()
    low = df['low'].values.copy()
    
    pivothigh_idx = np.full(n, -1, dtype=np.int32)
    pivotlow_idx = np.full(n, -1, dtype=np.int32)
    
    for i in range(pp, n - pp):
        is_high = True
        for j in range(1, pp + 1):
            if high[i] <= high[i - j] or high[i] < high[i + j]:
                is_high = False
                break
        if is_high:
            pivothigh_idx[i] = i
        
        is_low = True
        for j in range(1, pp + 1):
            if low[i] >= low[i - j] or low[i] > low[i + j]:
                is_low = False
                break
        if is_low:
            pivotlow_idx[i] = i
    
    zigzag_type = np.full(n, '', dtype=object)
    zigzag_val = np.full(n, np.nan, dtype=np.float64)
    zigzag_idx = np.full(n, -1, dtype=np.int32)
    
    pivot_indices = np.sort(np.concatenate([
        np.where(pivothigh_idx >= 0)[0],
        np.where(pivotlow_idx >= 0)[0]
    ]))
    
    if len(pivot_indices) > 0:
        zigzag_idx[pivot_indices[0]] = pivot_indices[0]
        if pivothigh_idx[pivot_indices[0]] >= 0:
            zigzag_type[pivot_indices[0]] = 'H'
            zigzag_val[pivot_indices[0]] = high[pivot_indices[0]]
        else:
            zigzag_type[pivot_indices[0]] = 'L'
            zigzag_val[pivot_indices[0]] = low[pivot_indices[0]]
        
        for k in range(1, len(pivot_indices)):
            idx = pivot_indices[k]
            zigzag_idx[idx] = idx
            prev_idx = pivot_indices[k - 1]
            
            if pivothigh_idx[idx] >= 0:
                zigzag_type[idx] = 'H'
                zigzag_val[idx] = high[idx]
            else:
                zigzag_type[idx] = 'L'
                zigzag_val[idx] = low[idx]
    
    major_high_level = np.nan
    major_low_level = np.nan
    major_high_idx = -1
    major_low_idx = -1
    major_high_type = ''
    major_low_type = ''
    
    dbTradeTriggered = False
    dtTradeTriggered = False
    isLongOpen = False
    isShortOpen = False
    
    entries = []
    trade_num = 1
    
    for i in range(1, n):
        if zigzag_idx[i] < 0:
            continue
        
        cur_type = zigzag_type[zigzag_idx[i]]
        cur_val = zigzag_val[zigzag_idx[i]]
        cur_idx = zigzag_idx[i]
        
        if cur_type == 'H':
            if not np.isnan(major_low_level):
                if cur_val > major_high_level:
                    major_high_level = cur_val
                    major_high_idx = cur_idx
                    major_high_type = 'HH'
                elif cur_val > major_low_level:
                    major_high_level = cur_val
                    major_high_idx = cur_idx
                    major_high_type = 'LH'
        elif cur_type == 'L':
            if not np.isnan(major_high_level):
                if cur_val < major_low_level:
                    major_low_level = cur_val
                    major_low_idx = cur_idx
                    major_low_type = 'LL'
                elif cur_val < major_high_level:
                    major_low_level = cur_val
                    major_low_idx = cur_idx
                    major_low_type = 'HL'
        
        if cur_type == 'H' and (major_high_type == 'HH' or major_high_type == 'LH'):
            major_high_level = cur_val
            major_high_idx = cur_idx
            if zigzag_idx[i - 1] >= 0 and zigzag_type[zigzag_idx[i - 1]] == 'L':
                prev_val = zigzag_val[zigzag_idx[i - 1]]
                if cur_val > prev_val:
                    major_high_type = 'HH'
                else:
                    major_high_type = 'LH'
        
        if cur_type == 'L' and (major_low_type == 'LL' or major_low_type == 'HL'):
            major_low_level = cur_val
            major_low_idx = cur_idx
            if zigzag_idx[i - 1] >= 0 and zigzag_type[zigzag_idx[i - 1]] == 'H':
                prev_val = zigzag_val[zigzag_idx[i - 1]]
                if cur_val < prev_val:
                    major_low_type = 'LL'
                else:
                    major_low_type = 'HL'
        
        if cur_type == 'H' and major_low_type in ['LL', 'HL']:
            major_high_level = cur_val
            major_high_idx = cur_idx
            major_high_type = 'HH'
        
        if cur_type == 'L' and major_high_type in ['HH', 'LH']:
            major_low_level = cur_val
            major_low_idx = cur_idx
            major_low_type = 'LL'
        
        if cur_type == 'L' and not isLongOpen and not isShortOpen:
            dbTradeTriggered = True
        if cur_type == 'H' and not isShortOpen and not isLongOpen:
            dtTradeTriggered = True
        
        if dbTradeTriggered and major_high_type in ['HH', 'LH']:
            entry_price = df['close'].iloc[i]
            entry_ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(entry_ts),
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            
            trade_num += 1
            isLongOpen = True
            dbTradeTriggered = False
        
        if dtTradeTriggered and major_low_type in ['LL', 'HL']:
            entry_price = df['close'].iloc[i]
            entry_ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(entry_ts),
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            
            trade_num += 1
            isShortOpen = True
            dtTradeTriggered = False
        
        if isLongOpen and cur_type == 'H':
            isLongOpen = False
        if isShortOpen and cur_type == 'L':
            isShortOpen = False
    
    return entries