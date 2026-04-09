import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    swing_length = 3
    
    high = df['high']
    low = df['low']
    close = df['close']
    time = df['time']
    
    # isSwingHigh: high[i] > high[i+1] and high[i] > high[i-1] and high[i+1] > high[i+2] and high[i-1] > high[i-2]
    h0 = high.shift(0)
    h1 = high.shift(1)
    h2 = high.shift(2)
    h_1 = high.shift(-1)
    h_2 = high.shift(-2)
    is_swing_high = (h0 > h_1) & (h0 > h1) & (h_1 > h_2) & (h1 > h2)
    
    # isSwingLow: low[i] < low[i+1] and low[i] < low[i-1] and low[i+1] < low[i+2] and low[i-1] < low[i-2]
    l0 = low.shift(0)
    l1 = low.shift(1)
    l2 = low.shift(2)
    l_1 = low.shift(-1)
    l_2 = low.shift(-2)
    is_swing_low = (l0 < l_1) & (l0 < l1) & (l_1 < l2) & (l1 < l2)
    
    entries = []
    trade_num = 1
    
    last_high = np.nan
    last_low = np.nan
    current_swing_high = np.nan
    current_swing_low = np.nan
    
    for i in range(len(df)):
        # Update current swing points
        if is_swing_high.iloc[i]:
            current_swing_high = high.iloc[i]
        else:
            current_swing_high = np.nan
            
        if is_swing_low.iloc[i]:
            current_swing_low = low.iloc[i]
        else:
            current_swing_low = np.nan
        
        # Track last high and low (BOS logic)
        if not pd.isna(current_swing_high):
            last_high = current_swing_high
        if not pd.isna(current_swing_low):
            last_low = current_swing_low
        
        # Check for CHOCH conditions and emit entries
        # Bullish CHOCH: swing low > last high -> LONG
        if pd.notna(current_swing_low) and pd.notna(last_high):
            if current_swing_low > last_high:
                ts = time.iloc[i]
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(ts),
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': close.iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': current_swing_low,
                    'raw_price_b': current_swing_low
                })
                trade_num += 1
        
        # Bearish CHOCH: swing high < last low -> SHORT
        if pd.notna(current_swing_high) and pd.notna(last_low):
            if current_swing_high < last_low:
                ts = time.iloc[i]
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(ts),
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': close.iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': current_swing_high,
                    'raw_price_b': current_swing_high
                })
                trade_num += 1
    
    return entries