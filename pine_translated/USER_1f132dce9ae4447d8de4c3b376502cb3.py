import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.sort_values('time').reset_index(drop=True)
    
    length = 5
    useCloseCandle = False
    tradetype = "Long and Short"
    
    window = length * 2 + 1
    
    h_4h_arr = np.full(len(df), np.nan)
    l_4h_arr = np.full(len(df), np.nan)
    
    for i in range(window - 1, len(df)):
        h_4h_arr[i] = df['high'].iloc[i - window + 1:i + 1].max()
        l_4h_arr[i] = df['low'].iloc[i - window + 1:i + 1].min()
    
    h_4h = pd.Series(h_4h_arr, index=df.index)
    l_4h = pd.Series(l_4h_arr, index=df.index)
    
    lastHigh_arr = np.zeros(len(df))
    lastLow_arr = np.zeros(len(df))
    dirUp_arr = np.zeros(len(df), dtype=bool)
    
    for i in range(window, len(df)):
        h = h_4h_arr[i]
        l = l_4h_arr[i]
        
        if i > 0:
            dirUp = dirUp_arr[i-1]
            lastHigh = lastHigh_arr[i-1]
            lastLow = lastLow_arr[i-1]
        else:
            dirUp = False
            lastHigh = 0.0
            lastLow = 1e10
        
        isMax = (h == df['high'].iloc[max(0, i-length):i+length+1].max())
        isMin = (l == df['low'].iloc[max(0, i-length):i+length+1].min())
        
        if dirUp:
            if isMin and l < lastLow:
                lastLow = l
            if isMax and h > lastLow:
                lastHigh = h
                dirUp = False
        else:
            if isMax and h > lastHigh:
                lastHigh = h
            if isMin and l < lastHigh:
                lastLow = l
                dirUp = True
                if isMax and h > lastLow:
                    lastHigh = h
                    dirUp = False
        
        lastHigh_arr[i] = lastHigh
        lastLow_arr[i] = lastLow
        dirUp_arr[i] = dirUp
    
    lastHigh_series = pd.Series(lastHigh_arr, index=df.index)
    lastLow_series = pd.Series(lastLow_arr, index=df.index)
    
    recent_touch = pd.Series(False, index=df.index)
    for i in range(window + 1, len(df)):
        for j in range(1, 11):
            if i - j >= 0 and i - j - 1 >= 0:
                l_4h_prev = l_4h_arr[i - j]
                l_4h_curr = l_4h_arr[i - j - 1]
                ll_prev = lastLow_arr[i - j]
                h_4h_prev = h_4h_arr[i - j]
                h_4h_curr = h_4h_arr[i - j - 1]
                lh_prev = lastHigh_arr[i - j]
                if (l_4h_prev <= ll_prev and l_4h_curr > ll_prev) or (h_4h_prev >= lh_prev and h_4h_curr < lh_prev):
                    recent_touch.iloc[i] = True
                    break
    
    close = df['close']
    
    src_long = close if useCloseCandle else h_4h
    src_short = close if useCloseCandle else l_4h
    
    long_condition = (src_long >= lastHigh_series) & (h_4h.shift(1) < lastHigh_series.shift(1)) & (~recent_touch)
    short_condition = (src_short <= lastLow_series) & (l_4h.shift(1) > lastLow_series.shift(1)) & (~recent_touch)
    
    if tradetype == "Long":
        short_condition[:] = False
    elif tradetype == "Short":
        long_condition[:] = False
    
    entries = []
    trade_num = 1
    
    for i in range(window + 1, len(df)):
        entry_price = df['close'].iloc[i]
        ts = int(df['time'].iloc[i])
        entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        
        if long_condition.iloc[i] and not short_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
        elif short_condition.iloc[i] and not long_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
    
    return entries