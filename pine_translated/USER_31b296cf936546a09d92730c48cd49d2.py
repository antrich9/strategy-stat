import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    prev_day_high = np.nan
    prev_day_low = np.nan
    swept_high = False
    swept_low = False
    found_fvg = False
    first_sweep_taken = False
    
    entries = []
    trade_num = 1
    
    # Previous day's OHLC (for shift operations)
    prev_day_high_arr = df['high'].shift(1)
    prev_day_low_arr = df['low'].shift(1)
    close_shifted = df['close'].shift(1)
    
    # Detect new trading day (first bar of a new day based on date change)
    dates = pd.to_datetime(df['time'], unit='s', utc=True).dt.date
    new_day = dates != dates.shift(1)
    
    for i in range(len(df)):
        # Update previous day high/low at the start of new day
        if new_day.iloc[i] and i > 0:
            prev_day_high = df['high'].iloc[i-1]
            prev_day_low = df['low'].iloc[i-1]
            swept_high = False
            swept_low = False
            found_fvg = False
            first_sweep_taken = False
        
        # Bullish FVG: current low > high 2 bars back AND prior low > high 2 bars back
        if i >= 2:
            bull_fvg = (df['low'].iloc[i] > df['high'].iloc[i-2]) and (df['low'].iloc[i-1] > df['high'].iloc[i-2])
        else:
            bull_fvg = False
        
        # Bearish FVG: current high < low 2 bars back AND prior high < low 2 bars back
        if i >= 2:
            bear_fvg = (df['high'].iloc[i] < df['low'].iloc[i-2]) and (df['high'].iloc[i-1] < df['low'].iloc[i-2])
        else:
            bear_fvg = False
        
        # Check for sweeps of previous day's levels
        if not np.isnan(prev_day_high) and not swept_high and df['high'].iloc[i] > prev_day_high:
            swept_high = True
        
        if not np.isnan(prev_day_low) and not swept_low and df['low'].iloc[i] < prev_day_low:
            swept_low = True
        
        # Entry conditions: sweep occurred and FVG detected, not yet taken
        if (swept_high or swept_low) and bull_fvg and not found_fvg and not first_sweep_taken:
            found_fvg = True
            first_sweep_taken = True
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_idx': i,
                'entry_price': df['close'].iloc[i],
                'entry_time': df['time'].iloc[i]
            })
            trade_num += 1
        
        if (swept_high or swept_low) and bear_fvg and not found_fvg and not first_sweep_taken:
            found_fvg = True
            first_sweep_taken = True
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_idx': i,
                'entry_price': df['close'].iloc[i],
                'entry_time': df['time'].iloc[i]
            })
            trade_num += 1
    
    return entries