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
    entries = []
    trade_num = 1
    
    # Time window checks (London time: 08:00-09:45 and 15:00-16:45)
    df['ts_dt'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    df['hour'] = df['ts_dt'].dt.hour
    df['minute'] = df['ts_dt'].dt.minute
    
    # Morning window: 08:00 - 09:45
    morning_start = (df['hour'] == 8) & (df['minute'] >= 0)
    morning_end = (df['hour'] == 9) & (df['minute'] <= 45)
    isWithinMorningWindow = morning_start | morning_end
    
    # Afternoon window: 15:00 - 16:45
    afternoon_start = (df['hour'] == 15) & (df['minute'] >= 0)
    afternoon_end = (df['hour'] == 16) & (df['minute'] <= 45)
    isWithinAfternoonWindow = afternoon_start | afternoon_end
    
    in_trading_window = isWithinMorningWindow | isWithinAfternoonWindow
    
    # Helper functions for OB and FVG
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    close_arr = df['close'].values
    
    isUp = close_arr > open_arr
    isDown = close_arr < open_arr
    
    # OB functions (using shift for previous bar access)
    isObUp = np.zeros(len(df), dtype=bool)
    isObDown = np.zeros(len(df), dtype=bool)
    
    for i in range(2, len(df)):
        prev_idx = i - 1
        curr_idx = i
        if isDown[prev_idx] and isUp[curr_idx] and close_arr[curr_idx] > high_arr[prev_idx]:
            isObUp[i] = True
        if isUp[prev_idx] and isDown[curr_idx] and close_arr[curr_idx] < low_arr[prev_idx]:
            isObDown[i] = True
    
    # FVG functions
    isFvgUp = np.zeros(len(df), dtype=bool)
    isFvgDown = np.zeros(len(df), dtype=bool)
    
    for i in range(2, len(df)):
        if low_arr[i] > high_arr[i - 2]:
            isFvgUp[i] = True
        if high_arr[i] < low_arr[i - 2]:
            isFvgDown[i] = True
    
    obUp = isObUp
    obDown = isObDown
    fvgUp = isFvgUp
    fvgDown = isFvgDown
    
    # ATR filter implementation
    tr = np.maximum(
        np.maximum(
            high_arr - low_arr,
            np.abs(high_arr - np.roll(close_arr, 1))
        ),
        np.abs(low_arr - np.roll(close_arr, 1))
    )
    tr[0] = 0
    
    atr = np.zeros(len(df))
    alpha = 1.0 / 20.0
    atr[0] = tr[0]
    for i in range(1, len(df)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
    
    atr_value = atr / 1.5
    
    # Volume filter
    vol_sma = pd.Series(df['volume'].values).rolling(9).mean().values
    volfilt = df['volume'].shift(1).values > vol_sma * 1.5
    
    # ATR filter
    atrfilt = (df['low'].values - np.roll(high_arr, 2) > atr_value) | (np.roll(low_arr, 2) - high_arr > atr_value)
    atrfilt[0] = True
    atrfilt[1] = True
    
    # Trend filter
    loc = pd.Series(close_arr).ewm(span=54, adjust=False).mean().values
    loc2 = loc > np.roll(loc, 1)
    loc2[0] = False
    locfiltb = loc2
    locfilts = ~loc2
    
    # Breakaway FVG conditions
    bfvg = (low_arr > np.roll(high_arr, 2)) & volfilt & atrfilt & locfiltb
    sfvg = (high_arr < np.roll(low_arr, 2)) & volfilt & atrfilt & locfilts
    
    # Entry conditions
    long_condition = (obUp | fvgUp | bfvg) & in_trading_window.values
    short_condition = (obDown | fvgDown | sfvg) & in_trading_window.values
    
    # Generate entries
    for i in range(len(df)):
        if i < 2:
            continue
            
        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
        price = float(df['close'].iloc[i])
        
        if long_condition.iloc[i] if hasattr(long_condition, 'iloc') else long_condition[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1
        elif short_condition.iloc[i] if hasattr(short_condition, 'iloc') else short_condition[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1
    
    return entries