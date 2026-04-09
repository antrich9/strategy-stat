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
    
    df = df.copy()
    df['ts'] = df['time']
    
    # Detect NY midnight session start (hour == 0 and minute == 0)
    dt = pd.to_datetime(df['ts'], unit='s', utc=True)
    df['hour'] = dt.dt.hour
    df['minute'] = dt.dt.minute
    df['is_new_session'] = (df['hour'] == 0) & (df['minute'] == 0)
    
    # Calculate NY Midnight High/Low with sweep flags
    nyMidnightHigh = pd.Series(np.nan, index=df.index)
    nyMidnightLow = pd.Series(np.nan, index=df.index)
    flagNYHigh = pd.Series(False, index=df.index)
    flagNYLow = pd.Series(False, index=df.index)
    
    current_high = np.nan
    current_low = np.nan
    high_swept = False
    low_swept = False
    should_reset = False
    
    for i in range(len(df)):
        if df['is_new_session'].iloc[i]:
            current_high = np.nan
            current_low = np.nan
            high_swept = False
            low_swept = False
            should_reset = False
        
        if np.isnan(current_high):
            current_high = df['high'].iloc[i]
        else:
            current_high = max(current_high, df['high'].iloc[i])
        
        if np.isnan(current_low):
            current_low = df['low'].iloc[i]
        else:
            current_low = min(current_low, df['low'].iloc[i])
        
        nyMidnightHigh.iloc[i] = current_high
        nyMidnightLow.iloc[i] = current_low
        
        if should_reset:
            high_swept = False
            low_swept = False
            should_reset = False
        
        if df['close'].iloc[i] > current_high and not high_swept:
            flagNYHigh.iloc[i] = True
            high_swept = True
        
        if df['close'].iloc[i] < current_low and not low_swept:
            flagNYLow.iloc[i] = True
            low_swept = True
        
        if high_swept and low_swept:
            should_reset = True
    
    # OB and FVG conditions
    is_up = df['close'] > df['open']
    is_down = df['close'] < df['open']
    
    ob_up = is_down.shift(1) & is_up & (df['close'] > df['high'].shift(1))
    fvg_up = df['low'] > df['high'].shift(2)
    
    # Bullish stacked OB + FVG
    bull_stack = ob_up & fvg_up
    
    # Entries: NYLow swept + Bullish stacked OB+FVG
    entry_cond = flagNYLow & bull_stack
    
    # Build entries list
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if i < 3:
            continue
        if entry_cond.iloc[i]:
            entry_ts = int(df['ts'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
            
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
    
    return entries