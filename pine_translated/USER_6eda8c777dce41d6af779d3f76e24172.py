import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    timestamps = df['time'].values
    n = len(df)
    
    # Parameters
    PP = 5
    atrLength = 7
    atrMultiplier = 3.0
    
    # True Range
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hpc = abs(highs[i] - closes[i-1])
        lpc = abs(lows[i] - closes[i-1])
        tr[i] = max(hl, max(hpc, lpc))
    
    # ATR (Wilder's method)
    atr = np.zeros(n)
    if n >= atrLength:
        atr[atrLength-1] = np.mean(tr[:atrLength])
        for i in range(atrLength, n):
            atr[i] = (atr[i-1] * (atrLength - 1) + tr[i]) / atrLength
    
    # Pivot detection
    swing_high = np.zeros(n)
    swing_low = np.zeros(n)
    
    for i in range(PP, n - PP):
        if highs[i] == max(highs[i-PP:i+PP+1]):
            swing_high[i] = highs[i]
        if lows[i] == min(lows[i-PP:i+PP+1]):
            swing_low[i] = lows[i]
    
    # Track pivots for ZigZag
    pivot_types = []
    pivot_values = []
    pivot_indices = []
    
    for i in range(n):
        if swing_high[i] > 0:
            pivot_types.append('H')
            pivot_values.append(swing_high[i])
            pivot_indices.append(i)
        elif swing_low[i] > 0:
            pivot_types.append('L')
            pivot_values.append(swing_low[i])
            pivot_indices.append(i)
    
    # Major levels
    major_high = np.full(n, np.nan)
    major_low = np.full(n, np.nan)
    minor_high = np.full(n, np.nan)
    minor_low = np.full(n, np.nan)
    
    # BoS and ChoCh flags
    bullish_maj_bos = np.zeros(n, dtype=bool)
    bearish_maj_bos = np.zeros(n, dtype=bool)
    bullish_maj_choch = np.zeros(n, dtype=bool)
    bearish_maj_choch = np.zeros(n, dtype=bool)
    
    # Process pivots to identify structure
    for idx in range(len(pivot_types)):
        if idx < 2:
            continue
        
        # Major structure detection
        # Look at last 4 pivots for structure
        if idx >= 3:
            p0_type = pivot_types[idx-3] if idx-3 >= 0 else ''
            p1_type = pivot_types[idx-2] if idx-2 >= 0 else ''
            p2_type = pivot_types[idx-1] if idx-1 >= 0 else ''
            p3_type = pivot_types[idx]
            
            p1_val = pivot_values[idx-2] if idx-2 >= 0 else 0
            p2_val = pivot_values[idx-1] if idx-1 >= 0 else 0
            p3_val = pivot_values[idx]
            
            p1_idx = pivot_indices[idx-2] if idx-2 >= 0 else 0
            p2_idx = pivot_indices[idx-1] if idx-1 >= 0 else 0
            p3_idx = pivot_indices[idx]
            
            if p2_type == 'H' and p3_type == 'L':
                if p2_val > p1_val and p3_val < p2_val:
                    bearish_maj_bos[p3_idx] = True
            elif p2_type == 'L' and p3_type == 'H':
                if p2_val < p1_val and p3_val > p2_val:
                    bullish_maj_bos[p3_idx] = True
    
    # Update major levels using pivots
    high_pivots_idx = [i for i, t in enumerate(pivot_types) if t == 'H']
    low_pivots_idx = [i for i, t in enumerate(pivot_types) if t == 'L']
    
    for i in range(n):
        recent_high_idx = [idx for idx in high_pivots_idx if idx <= i]
        recent_low_idx = [idx for idx in low_pivots_idx if idx <= i]
        
        if len(recent_high_idx) >= 2:
            last_two = recent_high_idx[-2:]
            major_high[i] = max(pivot_values[j] for j in last_two)
        elif len(recent_high_idx) == 1:
            major_high[i] = pivot_values[recent_high_idx[0]]
            
        if len(recent_low_idx) >= 2:
            last_two = recent_low_idx[-2:]
            major_low[i] = min(pivot_values[j] for j in last_two)
        elif len(recent_low_idx) == 1:
            major_low[i] = pivot_values[recent_low_idx[0]]
    
    # Entry signals
    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        if np.isnan(major_high[i]) or np.isnan(major_low[i]):
            continue
            
        # Bullish: Close breaks above major high
        if closes[i] > major_high[i] and closes[i-1] <= major_high[i]:
            long_entry[i] = True
            
        # Bearish: Close breaks below major low
        if closes[i] < major_low[i] and closes[i-1] >= major_low[i]:
            short_entry[i] = True
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(n):
        if long_entry[i]:
            entry_price = closes[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(timestamps[i]),
                'entry_time': datetime.fromtimestamp(timestamps[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif short_entry[i]:
            entry_price = closes[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(timestamps[i]),
                'entry_time': datetime.fromtimestamp(timestamps[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries