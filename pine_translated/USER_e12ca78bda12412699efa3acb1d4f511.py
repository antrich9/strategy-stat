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
    trade_num = 0
    
    n = len(df)
    if n < 5:
        return results
    
    open_vals = df['open'].values
    high_vals = df['high'].values
    low_vals = df['low'].values
    close_vals = df['close'].values
    time_vals = df['time'].values
    
    asia_session_start_hour = 23
    asia_session_end_hour = 7
    
    asiaHigh = np.nan
    asiaLow = np.nan
    inAsiaSession = False
    inAsiaSession_prev = False
    
    # Calculate Asia session high/low for each bar
    asiaHigh_arr = np.full(n, np.nan)
    asiaLow_arr = np.full(n, np.nan)
    
    for i in range(n):
        ts = time_vals[i]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        
        inAsiaSession_prev = inAsiaSession
        
        if hour == asia_session_start_hour:
            inAsiaSession = True
        elif hour == asia_session_end_hour and inAsiaSession:
            inAsiaSession = False
        
        if inAsiaSession:
            if np.isnan(asiaHigh):
                asiaHigh = high_vals[i]
                asiaLow = low_vals[i]
            else:
                asiaHigh = max(asiaHigh, high_vals[i])
                asiaLow = min(asiaLow, low_vals[i])
        
        asiaHigh_arr[i] = asiaHigh
        asiaLow_arr[i] = asiaLow
    
    london_start_morning_hour = 6
    london_start_morning_min = 45
    london_end_morning_hour = 9
    london_end_morning_min = 45
    
    london_start_afternoon_hour = 14
    london_start_afternoon_min = 45
    london_end_afternoon_hour = 16
    london_end_afternoon_min = 45
    
    isWithinTimeWindow = np.zeros(n, dtype=bool)
    
    for i in range(n):
        ts = time_vals[i]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        
        morning_start = hour * 60 + minute >= london_start_morning_hour * 60 + london_start_morning_min
        morning_end = hour * 60 + minute < london_end_morning_hour * 60 + london_end_morning_min
        
        afternoon_start = hour * 60 + minute >= london_start_afternoon_hour * 60 + london_start_afternoon_min
        afternoon_end = hour * 60 + minute < london_end_afternoon_hour * 60 + london_end_afternoon_min
        
        isWithinTimeWindow[i] = (morning_start and morning_end) or (afternoon_start and afternoon_end)
    
    isUp = np.zeros(n, dtype=bool)
    isDown = np.zeros(n, dtype=bool)
    for i in range(n):
        isUp[i] = close_vals[i] > open_vals[i]
        isDown[i] = close_vals[i] < open_vals[i]
    
    isObUp = np.zeros(n, dtype=bool)
    isObDown = np.zeros(n, dtype=bool)
    for i in range(2, n):
        if isDown[i-1] and isUp[i] and close_vals[i] > high_vals[i-1]:
            isObUp[i] = True
    
    for i in range(2, n):
        if isUp[i-1] and isDown[i] and close_vals[i] < low_vals[i-1]:
            isObDown[i] = True
    
    fvgUp = np.zeros(n, dtype=bool)
    fvgDown = np.zeros(n, dtype=bool)
    for i in range(2, n):
        if low_vals[i] > high_vals[i-2]:
            fvgUp[i] = True
    
    for i in range(2, n):
        if high_vals[i] < low_vals[i-2]:
            fvgDown[i] = True
    
    asiahighSwept = np.zeros(n, dtype=bool)
    asialowSwept = np.zeros(n, dtype=bool)
    for i in range(n):
        if not np.isnan(asiaHigh_arr[i]) and high_vals[i] > asiaHigh_arr[i]:
            asiahighSwept[i] = True
        if not np.isnan(asiaLow_arr[i]) and low_vals[i] < asiaLow_arr[i]:
            asialowSwept[i] = True
    
    obUp = isObUp
    obDown = isObDown
    fvgUp_vals = fvgUp
    fvgDown_vals = fvgDown
    
    for i in range(3, n):
        if isWithinTimeWindow[i]:
            bullish_entry = asialowSwept[i] and obUp[i] and fvgUp_vals[i]
            bearish_entry = asiahighSwept[i] and obDown[i] and fvgDown_vals[i]
            
            if bullish_entry:
                trade_num += 1
                entry_ts = int(time_vals[i])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                entry_price = float(close_vals[i])
                
                results.append({
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
            
            if bearish_entry:
                trade_num += 1
                entry_ts = int(time_vals[i])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                entry_price = float(close_vals[i])
                
                results.append({
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
    
    return results