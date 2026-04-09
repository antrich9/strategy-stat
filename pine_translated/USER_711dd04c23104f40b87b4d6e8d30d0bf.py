import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Extract OHLC data
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    opens = df['open'].values
    timestamps = df['time'].values
    n = len(df)
    
    # Parameters from Pine Script
    PP = 5  # Pivot Period
    atr_length = 55  # ATR length from script
    atr_mult = 1.0  # ATR multiplier (typically 1.0, used for structure validation)
    
    # Calculate ATR using Wilder's method (ta.atr)
    tr1 = highs - lows
    tr2 = np.abs(highs - np.roll(opens, 1))
    tr3 = np.abs(lows - np.roll(opens, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar True Range is High-Low
    
    atr = np.zeros(n)
    if n >= atr_length:
        atr[atr_length-1] = np.mean(tr[:atr_length])
        for i in range(atr_length, n):
            atr[i] = (atr[i-1] * (atr_length - 1) + tr[i]) / atr_length
    else:
        return []
    
    # Find pivot highs using rolling window (current bar is highest in last PP bars)
    pivot_high_idx = np.zeros(n, dtype=int)
    for i in range(PP, n):
        window_start = i - PP
        if closes[i] >= np.max(closes[window_start:i]):
            pivot_high_idx[i] = i
    
    pivot_high_idx[pivot_high_idx == 0] = -1
    pivot_high_idx[pivot_high_idx > 0] -= 1
    
    # Find pivot lows using rolling window (current bar is lowest in last PP bars)
    pivot_low_idx = np.zeros(n, dtype=int)
    for i in range(PP, n):
        window_start = i - PP
        if closes[i] <= np.min(closes[window_start:i]):
            pivot_low_idx[i] = i
    
    pivot_low_idx[pivot_low_idx == 0] = -1
    pivot_low_idx[pivot_low_idx > 0] -= 1
    
    # Extract actual high/low values at pivot indices
    pivot_high_vals = np.where(pivot_high_idx >= 0, highs[pivot_high_idx], np.nan)
    pivot_low_vals = np.where(pivot_low_idx >= 0, lows[pivot_low_idx], np.nan)
    
    # Major structure tracking variables
    major_high = np.nan
    major_low = np.nan
    major_high_idx = 0
    major_low_idx = 0
    prev_major_high = np.nan
    prev_major_low = np.nan
    
    higher_hhs = 0  # Consecutive Higher Highs
    lower_lls = 0   # Consecutive Lower Lows
    last_pivot_type = ""
    
    entries = []
    trade_num = 1
    
    for i in range(PP, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
        
        # Update major high
        if pivot_high_idx[i] >= 0:
            new_high = pivot_high_vals[i]
            if np.isnan(major_high) or new_high > major_high:
                major_high = new_high
                major_high_idx = pivot_high_idx[i]
        
        # Update major low
        if pivot_low_idx[i] >= 0:
            new_low = pivot_low_vals[i]
            if np.isnan(major_low) or new_low < major_low:
                major_low = new_low
                major_low_idx = pivot_low_idx[i]
        
        # BoS detection logic
        bos_bull = False
        bos_bear = False
        bullish_choch = False
        bearish_choch = False
        
        # Higher High logic
        if pivot_high_idx[i] >= 0:
            if not np.isnan(major_high) and not np.isnan(prev_major_high):
                if major_high > prev_major_high:
                    higher_hhs += 1
                    last_pivot_type = "HH"
                else:
                    higher_hhs = 0
                    last_pivot_type = ""
                
                if higher_hhs >= 2:
                    bos_bull = True
                
                if last_pivot_type == "HL" and pivot_high_vals[i] < major_high:
                    bullish_choch = True
            
            prev_major_high = major_high
        
        # Lower Low logic
        if pivot_low_idx[i] >= 0:
            if not np.isnan(major_low) and not np.isnan(prev_major_low):
                if major_low < prev_major_low:
                    lower_lls += 1
                    last_pivot_type = "LL"
                else:
                    lower_lls = 0
                    last_pivot_type = ""
                
                if lower_lls >= 2:
                    bos_bear = True
                
                if last_pivot_type == "LH" and pivot_low_vals[i] > major_low:
                    bearish_choch = True
            
            prev_major_low = major_low
        
        # Entry signals based on BoS and ChoCh detection
        if bos_bull or bullish_choch:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(timestamps[i]),
                'entry_time': datetime.fromtimestamp(timestamps[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(closes[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(closes[i]),
                'raw_price_b': float(closes[i])
            })
            trade_num += 1
        
        if bos_bear or bearish_choch:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(timestamps[i]),
                'entry_time': datetime.fromtimestamp(timestamps[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(closes[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(closes[i]),
                'raw_price_b': float(closes[i])
            })
            trade_num += 1
    
    return entries