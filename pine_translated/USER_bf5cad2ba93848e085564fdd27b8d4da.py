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
    
    results = []
    trade_num = 1
    
    if len(df) < 20:
        return results
    
    close = df['close']
    high = df['high']
    low = df['low']
    time = df['time']
    
    # Calculate EMAs (21 and 50 period as commonly used in ICT strategies)
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    
    # Detect STH (Short-Term High) pivots - local highs
    sth_high = pd.Series(False, index=df.index)
    sth_low = pd.Series(False, index=df.index)
    
    for i in range(2, len(df) - 2):
        if pd.notna(high.iloc[i-2]) and pd.notna(high.iloc[i-1]) and pd.notna(high.iloc[i]) and pd.notna(high.iloc[i+1]) and pd.notna(high.iloc[i+2]):
            if high.iloc[i-1] > high.iloc[i-2] and high.iloc[i-1] > high.iloc[i] and high.iloc[i-1] >= high.iloc[i+1] and high.iloc[i-1] >= high.iloc[i+2]:
                sth_high.iloc[i-1] = True
                
            if low.iloc[i-1] < low.iloc[i-2] and low.iloc[i-1] < low.iloc[i] and low.iloc[i-1] <= low.iloc[i+1] and low.iloc[i-1] <= low.iloc[i+2]:
                sth_low.iloc[i-1] = True
    
    # Detect breaks of structure
    sth_high_break = pd.Series(False, index=df.index)
    sth_low_break = pd.Series(False, index=df.index)
    
    for i in range(1, len(df)):
        if sth_high.iloc[i-1]:
            prev_sth_idx = i - 1
            for j in range(i, len(df)):
                if close.iloc[j] > high.iloc[prev_sth_idx]:
                    sth_high_break.iloc[j] = True
                    break
                    
        if sth_low.iloc[i-1]:
            prev_stl_idx = i - 1
            for j in range(i, len(df)):
                if close.iloc[j] < low.iloc[prev_stl_idx]:
                    sth_low_break.iloc[j] = True
                    break
    
    # EMA crossover conditions
    ema_fast_above = ema21 > ema50
    ema_fast_below = ema21 < ema50
    ema_fast_above_prev = ema_fast_above.shift(1).fillna(False)
    ema_fast_below_prev = ema_fast_below.shift(1).fillna(False)
    
    ema_bullish_cross = ema_fast_above & ~ema_fast_above_prev
    ema_bearish_cross = ema_fast_below & ~ema_fast_below_prev
    
    # Long entry conditions: EMA bullish cross + bullish structure break or confirmed uptrend
    long_condition = ema_bullish_cross & (ema21 > ema50)
    
    # Short entry conditions: EMA bearish cross + bearish structure break or confirmed downtrend
    short_condition = ema_bearish_cross & (ema21 < ema50)
    
    # Track positions
    in_long = False
    in_short = False
    
    for i in range(1, len(df)):
        if pd.isna(ema21.iloc[i]) or pd.isna(ema50.iloc[i]):
            continue
            
        entry_price = close.iloc[i]
        entry_ts = int(time.iloc[i])
        entry_time_str = datetime.fromtimestamp(entry_ts / 1000 if entry_ts > 1e12 else entry_ts, tz=timezone.utc).isoformat()
        
        # Long entry signal
        if long_condition.iloc[i] and not in_long and not in_short:
            in_long = True
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
            continue
        
        # Short entry signal
        if short_condition.iloc[i] and not in_short and not in_long:
            in_short = True
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
            continue
        
        # Reset position flags on opposite EMA cross
        if ema_bearish_cross.iloc[i] and in_long:
            in_long = False
        if ema_bullish_cross.iloc[i] and in_short:
            in_short = False
    
    return results