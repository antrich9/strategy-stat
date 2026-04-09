import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    ... (docstring provided)
    """
    # Constants
    PP = 6  # Pivot Period from Pine Script
    
    # Calculate ATR (Wilder's method) - though not strictly needed for entries, it's in the code
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR (Wilder)
    atr = tr.ewm(alpha=1/55, adjust=False).mean()  # ta.atr(55)
    
    # Pivot High and Low detection
    # Using a loop for clarity, though could be optimized
    pivot_high = pd.Series(index=df.index, dtype=float)
    pivot_low = pd.Series(index=df.index, dtype=float)
    
    # We need to look back PP and forward PP. 
    # For bar i, it's a pivot high if high[i] > high[j] for all j in [i-PP, i+PP] and i != j.
    # Since we are processing all data, we can calculate this using a rolling window with center.
    # However, pandas rolling window with center has edge effects.
    # Let's use a simple loop for the relevant range.
    
    for i in range(PP, len(df) - PP):
        # Check High Pivot
        window_high = high.iloc[i-PP:i+PP+1]
        if high.iloc[i] == window_high.max():
            pivot_high.iloc[i] = high.iloc[i]
        
        # Check Low Pivot
        window_low = low.iloc[i-PP:i+PP+1]
        if low.iloc[i] == window_low.min():
            pivot_low.iloc[i] = low.iloc[i]
    
    # Now we need to track the Major Highs and Lows.
    # The Pine Script maintains a history of pivots and classifies them.
    # For simplicity, let's assume we track the last two swing highs and lows.
    
    last_swing_high = None
    last_swing_low = None
    prev_swing_high = None
    prev_swing_low = None
    
    entries = []
    trade_num = 1
    
    # We need to skip initial bars where we don't have enough data
    for i in range(PP * 2 + 1, len(df)):  # Start from a point where we have pivots
        # Update swing highs and lows
        # Find the most recent pivots
        # We can optimize by tracking indices, but for now, let's find them in the last few bars
        
        # Look back to find the last two swing highs
        # We can scan backwards from current bar
        temp_high_indices = []
        for j in range(i, i - 100, -1):  # Look back up to 100 bars
            if pd.notna(pivot_high.iloc[j]):
                temp_high_indices.append(j)
                if len(temp_high_indices) == 2:
                    break
        
        if len(temp_high_indices) >= 2:
            last_swing_high_idx = temp_high_indices[0]
            prev_swing_high_idx = temp_high_indices[1]
            last_swing_high = high.iloc[last_swing_high_idx]
            prev_swing_high = high.iloc[prev_swing_high_idx]
        elif len(temp_high_indices) == 1:
            last_swing_high_idx = temp_high_indices[0]
            last_swing_high = high.iloc[last_swing_high_idx]
            prev_swing_high = None
        else:
            last_swing_high = None
            prev_swing_high = None
            
        # Look back for last two swing lows
        temp_low_indices = []
        for j in range(i, i - 100, -1):
            if pd.notna(pivot_low.iloc[j]):
                temp_low_indices.append(j)
                if len(temp_low_indices) == 2:
                    break
        
        if len(temp_low_indices) >= 2:
            last_swing_low_idx = temp_low_indices[0]
            prev_swing_low_idx = temp_low_indices[1]
            last_swing_low = low.iloc[last_swing_low_idx]
            prev_swing_low = low.iloc[prev_swing_low_idx]
        elif len(temp_low_indices) == 1:
            last_swing_low_idx = temp_low_indices[0]
            last_swing_low = low.iloc[last_swing_low_idx]
            prev_swing_low = None
        else:
            last_swing_low = None
            prev_swing_low = None
        
        # Entry Logic: Break of Structure
        # Bullish BOS: Price breaks above the last swing high, and the last swing high is higher than the previous (HH)
        # Bearish BOS: Price breaks below the last swing low, and the last swing low is lower than the previous (LL)
        
        current_close = close.iloc[i]
        
        # Long Entry Condition (dtTradeTriggered equivalent)
        # We check if price crosses above the last swing high
        # And ensure it's a valid HH (Higher High)
        if last_swing_high is not None and prev_swing_high is not None:
            if last_swing_high > prev_swing_high:  # Higher High
                # Check crossover: previous close <= last_swing_high and current close > last_swing_high
                prev_close = close.iloc[i-1]
                if prev_close <= last_swing_high and current_close > last_swing_high:
                    entry_price = current_close
                    entry_ts = df['time'].iloc[i]
                    entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                    
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'long',
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
        
        # Short Entry Condition (dbTradeTriggered equivalent)
        # Bearish BOS: Price breaks below the last swing low, and it's a Lower Low
        if last_swing_low is not None and prev_swing_low is not None:
            if last_swing_low < prev_swing_low:  # Lower Low
                prev_close = close.iloc[i-1]
                if prev_close >= last_swing_low and current_close < last_swing_low:
                    entry_price = current_close
                    entry_ts = df['time'].iloc[i]
                    entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                    
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'short',
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