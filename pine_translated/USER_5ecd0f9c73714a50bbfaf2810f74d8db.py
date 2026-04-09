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
    
    close = df['close']
    high = df['high']
    low = df['low']
    open_vals = df['open']
    volume = df['volume']
    
    n = len(df)
    
    last_swing_high11 = np.nan
    last_swing_low11 = np.nan
    lastSwingType11 = "none"
    
    pdHigh = np.nan
    pdLow = np.nan
    tempHigh = np.nan
    tempLow = np.nan
    sweptHigh = False
    sweptLow = False
    
    swing_high11 = np.zeros(n)
    swing_low11 = np.zeros(n)
    is_swing_high11 = np.zeros(n, dtype=bool)
    is_swing_low11 = np.zeros(n, dtype=bool)
    
    dailyHigh11 = np.zeros(n)
    dailyLow11 = np.zeros(n)
    dailyClose11 = np.zeros(n)
    dailyOpen11 = np.zeros(n)
    
    prevDayHigh11 = np.zeros(n)
    prevDayLow11 = np.zeros(n)
    
    dailyHigh21 = np.zeros(n)
    dailyLow21 = np.zeros(n)
    dailyHigh22 = np.zeros(n)
    dailyLow22 = np.zeros(n)
    
    for i in range(5, n):
        day_ts = df['time'].iloc[i] - (df['time'].iloc[i] % 86400000)
        prev_day_ts = day_ts - 86400000
        
        for j in range(i, n):
            if df['time'].iloc[j] >= day_ts and df['time'].iloc[j] < day_ts + 86400000:
                dailyHigh11[i] = max(dailyHigh11[i], high.iloc[j]) if dailyHigh11[i] != 0 else high.iloc[j]
                dailyLow11[i] = min(dailyLow11[i], low.iloc[j]) if dailyLow11[i] != 0 else low.iloc[j]
                dailyClose11[i] = close.iloc[j]
                dailyOpen11[i] = open_vals.iloc[j]
        
        for j in range(i):
            if df['time'].iloc[j] >= prev_day_ts and df['time'].iloc[j] < prev_day_ts + 86400000:
                prevDayHigh11[i] = max(prevDayHigh11[i], high.iloc[j]) if prevDayHigh11[i] != 0 else high.iloc[j]
                prevDayLow11[i] = min(prevDayLow11[i], low.iloc[j]) if prevDayLow11[i] != 0 else low.iloc[j]
        
        if i >= 2:
            prev2_day_ts = prev_day_ts - 86400000
            for j in range(i):
                if df['time'].iloc[j] >= prev2_day_ts and df['time'].iloc[j] < prev2_day_ts + 86400000:
                    dailyHigh22[i] = max(dailyHigh22[i], high.iloc[j]) if dailyHigh22[i] != 0 else high.iloc[j]
                    dailyLow22[i] = min(dailyLow22[i], low.iloc[j]) if dailyLow22[i] != 0 else low.iloc[j]
        
        for j in range(i):
            if df['time'].iloc[j] >= day_ts - 86400000 and df['time'].iloc[j] < day_ts:
                dailyHigh21[i] = max(dailyHigh21[i], high.iloc[j]) if dailyHigh21[i] != 0 else high.iloc[j]
                dailyLow21[i] = min(dailyLow21[i], low.iloc[j]) if dailyLow21[i] != 0 else low.iloc[j]
    
    if n > 0:
        dailyHigh11[0] = high.iloc[0]
        dailyLow11[0] = low.iloc[0]
        dailyClose11[0] = close.iloc[0]
        dailyOpen11[0] = open_vals.iloc[0]
    
    for i in range(5, n):
        is_swing_high11[i] = dailyHigh21[i] < dailyHigh22[i] and dailyHigh11[i] < dailyHigh22[i] and (i >= 4 and dailyHigh11[i] < dailyHigh22[i])
        is_swing_low11[i] = dailyLow21[i] > dailyLow22[i] and dailyLow11[i] > dailyLow22[i] and (i >= 4 and dailyLow11[i] > dailyLow22[i])
        
        if i >= 3 and dailyHigh11[i] < dailyHigh22[i]:
            is_swing_high11[i] = is_swing_high11[i] and True
        if i >= 4 and dailyHigh11[i-3] < dailyHigh22[i]:
            is_swing_high11[i] = is_swing_high11[i] and True
        if i >= 5 and dailyHigh11[i-4] < dailyHigh22[i]:
            is_swing_high11[i] = is_swing_high11[i] and True
            
        if i >= 3 and dailyLow11[i] > dailyLow22[i]:
            is_swing_low11[i] = is_swing_low11[i] and True
        if i >= 4 and dailyLow11[i-3] > dailyLow22[i]:
            is_swing_low11[i] = is_swing_low11[i] and True
        if i >= 5 and dailyLow11[i-4] > dailyLow22[i]:
            is_swing_low11[i] = is_swing_low11[i] and True
            
        if is_swing_high11[i]:
            swing_high11[i] = dailyHigh22[i]
            last_swing_high11 = swing_high11[i]
            lastSwingType11 = "dailyHigh"
        if is_swing_low11[i]:
            swing_low11[i] = dailyLow22[i]
            last_swing_low11 = swing_low11[i]
            lastSwingType11 = "dailyLow"
    
    for i in range(1, n):
        current_day_ts = df['time'].iloc[i] - (df['time'].iloc[i] % 86400000)
        prev_bar_day_ts = df['time'].iloc[i-1] - (df['time'].iloc[i-1] % 86400000)
        newDay = current_day_ts != prev_bar_day_ts
        
        if newDay:
            pdHigh = tempHigh
            pdLow = tempLow
            tempHigh = high.iloc[i]
            tempLow = low.iloc[i]
            sweptHigh = False
            sweptLow = False
        else:
            tempHigh = high.iloc[i] if np.isnan(tempHigh) else np.maximum(tempHigh, high.iloc[i])
            tempLow = low.iloc[i] if np.isnan(tempLow) else np.minimum(tempLow, low.iloc[i])
        
        if pdHigh == pdHigh and high.iloc[i] > pdHigh and not sweptHigh:
            sweptHigh = True
        if pdLow == pdLow and low.iloc[i] < pdLow and not sweptLow:
            sweptLow = True
    
    london_start_window1 = 7 * 60
    london_end_window1 = (11 * 60 + 45)
    london_start_window2 = 14 * 60
    london_end_window2 = (14 * 60 + 45)
    
    high_4h = np.zeros(n)
    low_4h = np.zeros(n)
    close_4h = np.zeros(n)
    open_4h = np.zeros(n)
    volume_4h = np.zeros(n)
    
    high_4h_1 = np.zeros(n)
    low_4h_1 = np.zeros(n)
    high_4h_2 = np.zeros(n)
    low_4h_2 = np.zeros(n)
    close_4h_1 = np.zeros(n)
    
    for i in range(n):
        bar_ts = df['time'].iloc[i]
        minutes_since_midnight = (bar_ts % 86400000) // 60000
        hour = minutes_since_midnight // 60
        minute = minutes_since_midnight % 60
        total_minutes = hour * 60 + minute
        
        four_hour_bins = [0, 240, 480, 720, 960, 1200, 1440]
        current_4h_bin = 0
        for k in range(len(four_hour_bins) - 1):
            if total_minutes >= four_hour_bins[k] and total_minutes < four_hour_bins[k + 1]:
                current_4h_bin = four_hour_bins[k]
                break
        
        next_4h_bin = current_4h_bin + 240
        prev_4h_bin = current_4h_bin - 240
        prev2_4h_bin = current_4h_bin - 480
        
        current_4h_start = bar_ts - (bar_ts % 86400000) + current_4h_bin * 60000
        prev_4h_start = current_4h_start - 240 * 60000
        prev2_4h_start = current_4h_start - 480 * 60000
        next_4h_end = current_4h_start + 240 * 60000
        
        for j in range(i + 1, n):
            check_ts = df['time'].iloc[j]
            if check_ts >= current_4h_start and check_ts < next_4h_end:
                high_4h[i] = max(high_4h[i], high.iloc[j]) if high_4h[i] != 0 else high.iloc[j]
                low_4h[i] = min(low_4h[i], low.iloc[j]) if low_4h[i] != 0 else low.iloc[j]
                close_4h[i] = close.iloc[j]
                open_4h[i] = open_vals.iloc[j]
                volume_4h[i] = volume.iloc[j]
        
        for j in range(i + 1, n):
            check_ts = df['time'].iloc[j]
            if check_ts >= prev_4h_start and check_ts < current_4h_start:
                high_4h_1[i] = max(high_4h_1[i], high.iloc[j]) if high_4h_1[i] != 0 else high.iloc[j]
                low_4h_1[i] = min(low_4h_1[i], low.iloc[j]) if low_4h_1[i] != 0 else low.iloc[j]
        
        for j in range(i + 1, n):
            check_ts = df['time'].iloc[j]
            if check_ts >= prev2_4h_start and check_ts < prev_4h_start:
                high_4h_2[i] = max(high_4h_2[i], high.iloc[j]) if high_4h_2[i] != 0 else high.iloc[j]
                low_4h_2[i] = min(low_4h_2[i], low.iloc[j]) if low_4h_2[i] != 0 else low.iloc[j]
        
        for j in range(i + 1, n):
            check_ts = df['time'].iloc[j]
            if check_ts >= prev_4h_start and check_ts < current_4h_start:
                close_4h_1[i] = close.iloc[j]
    
    if n > 0:
        bar_ts = df['time'].iloc[0]
        minutes_since_midnight = (bar_ts % 86400000) // 60000
        hour = minutes_since_midnight // 60
        minute = minutes_since_midnight % 60
        total_minutes = hour * 60 + minute
        
        four_hour_bins = [0, 240, 480, 720, 960, 1200, 1440]
        current_4h_bin = 0
        for k in range(len(four_hour_bins) - 1):
            if total_minutes >= four_hour_bins[k] and total_minutes < four_hour_bins[k + 1]:
                current_4h_bin = four_hour_bins[k]
                break
        
        current_4h_start = bar_ts - (bar_ts % 86400000) + current_4h_bin * 60000
        next_4h_end = current_4h_start + 240 * 60000
        
        high_4h[0] = high.iloc[0]
        low_4h[0] = low.iloc[0]
        close_4h[0] = close.iloc[0]
        open_4h[0] = open_vals.iloc[0]
        volume_4h[0] = volume.iloc[0]
    
    loc1 = close_4h.rolling(54).mean()
    loc21 = loc1 > loc1.shift(1)
    
    fivg_bull = low_4h - high_4h_2
    fivg_bear = low_4h_2 - high_4h
    
    for i in range(1, n):
        bar_ts = df['time'].iloc[i]
        minutes_since_midnight = (bar_ts % 86400000) // 60000
        hour = minutes_since_midnight // 60
        minute = minutes_since_midnight % 60
        total_minutes = hour * 60 + minute
        
        in_window = (total_minutes >= london_start_window1 and total_minutes < london_end_window1) or \
                    (total_minutes >= london_start_window2 and total_minutes < london_end_window2)
        
        bfvg = low_4h.iloc[i] > high_4h_2.iloc[i]
        sfvg = high_4h.iloc[i] < low_4h_2.iloc[i]
        
        bull_entry = bfvg and in_window and lastSwingType11 == "dailyLow"
        bear_entry = sfvg and in_window and lastSwingType11 == "dailyHigh"
        
        if bull_entry:
            entry_price = close.iloc[i]
            entry_ts = int(i * 60000)
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            
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
            trade_num += 1
        
        if bear_entry:
            entry_price = close.iloc[i]
            entry_ts = int(i * 60000)
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            
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
            trade_num += 1
        
        if is_swing_high11[i]:
            lastSwingType11 = "dailyHigh"
        if is_swing_low11[i]:
            lastSwingType11 = "dailyLow"
        
        current_day_ts = df['time'].iloc[i] - (df['time'].iloc[i] % 86400000)
        prev_bar_day_ts = df['time'].iloc[i-1] - (df['time'].iloc[i-1] % 86400000)
        newDay = current_day_ts != prev_bar_day_ts
        
        if newDay:
            pdHigh = tempHigh
            pdLow = tempLow
            tempHigh = high.iloc[i]
            tempLow = low.iloc[i]
            sweptHigh = False
            sweptLow = False
        else:
            tempHigh = high.iloc[i] if np.isnan(tempHigh) else np.maximum(tempHigh, high.iloc[i])
            tempLow = low.iloc[i] if np.isnan(tempLow) else np.minimum(tempLow, low.iloc[i])
        
        if not np.isnan(pdHigh) and high.iloc[i] > pdHigh and not sweptHigh:
            sweptHigh = True
        if not np.isnan(pdLow) and low.iloc[i] < pdLow and not sweptLow:
            sweptLow = True
    
    return results