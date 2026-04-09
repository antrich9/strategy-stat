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
    
    left = 20
    right = 15
    atrLen = 30
    mult = 0.5
    
    results = []
    trade_num = 0
    
    close = df['close']
    high = df['high']
    low = df['low']
    
    # Identify pivot highs (swing highs)
    pivot_high = np.zeros(len(df))
    for i in range(left + right, len(df)):
        is_high = True
        for j in range(1, left + 1):
            if high.iloc[i - j] >= high.iloc[i]:
                is_high = False
                break
        if is_high:
            for j in range(1, right + 1):
                if high.iloc[i + j] > high.iloc[i] and (i + j) < len(df):
                    is_high = False
                    break
        if is_high:
            pivot_high[i] = high.iloc[i]
    
    # Identify pivot lows (swing lows)
    pivot_low = np.zeros(len(df))
    for i in range(left + right, len(df)):
        is_low = True
        for j in range(1, left + 1):
            if low.iloc[i - j] <= low.iloc[i]:
                is_low = False
                break
        if is_low:
            for j in range(1, right + 1):
                if low.iloc[i + j] < low.iloc[i] and (i + j) < len(df):
                    is_low = False
                    break
        if is_low:
            pivot_low[i] = low.iloc[i]
    
    # Wilder ATR calculation
    tr = np.maximum(high - low, np.maximum(
        np.abs(high - close.shift(1).fillna(0)),
        np.abs(low - close.shift(1).fillna(0))
    ))
    
    atr = pd.Series(tr).ewm(alpha=1.0/atrLen, adjust=False).mean()
    
    # Store recent pivot highs and lows for breakout detection
    recent_highs = []
    recent_lows = []
    
    for i in range(left + right + 1, len(df)):
        # Update recent pivots (keep last 4)
        if pivot_high[i] > 0:
            recent_highs.append(high.iloc[i])
            if len(recent_highs) > 4:
                recent_highs.pop(0)
        if pivot_low[i] > 0:
            recent_lows.append(low.iloc[i])
            if len(recent_lows) > 4:
                recent_lows.pop(0)
        
        if len(recent_highs) < 1 or len(recent_lows) < 1:
            continue
        
        current_high = high.iloc[i]
        current_low = low.iloc[i]
        current_close = close.iloc[i]
        current_atr = atr.iloc[i]
        
        if pd.isna(current_atr):
            continue
        
        zone_width = current_atr * mult
        
        # Find highest recent pivot and lowest recent pivot
        if len(recent_highs) > 0:
            resistance = max(recent_highs)
        else:
            resistance = current_high
        
        if len(recent_lows) > 0:
            support = min(recent_lows)
        else:
            support = current_low
        
        # Short entry: Breakdown below support zone
        if support > 0:
            if current_close.iloc[i] < (support - zone_width) and i > 0:
                prev_close = close.iloc[i - 1]
                if prev_close >= (support - zone_width):
                    trade_num += 1
                    entry_ts = int(df['time'].iloc[i])
                    entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                    results.append({
                        'trade_num': trade_num,
                        'direction': 'short',
                        'entry_ts': entry_ts,
                        'entry_time': entry_time,
                        'entry_price_guess': float(current_close),
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': float(current_close),
                        'raw_price_b': float(current_close)
                    })
        
        # Long entry: Breakout above resistance zone
        if resistance > 0:
            if current_close.iloc[i] > (resistance + zone_width) and i > 0:
                prev_close = close.iloc[i - 1]
                if prev_close <= (resistance + zone_width):
                    trade_num += 1
                    entry_ts = int(df['time'].iloc[i])
                    entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                    results.append({
                        'trade_num': trade_num,
                        'direction': 'long',
                        'entry_ts': entry_ts,
                        'entry_time': entry_time,
                        'entry_price_guess': float(current_close),
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': float(current_close),
                        'raw_price_b': float(current_close)
                    })
    
    return results