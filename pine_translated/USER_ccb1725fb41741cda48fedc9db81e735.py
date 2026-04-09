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
    
    n = len(df)
    if n < 3:
        return entries
    
    close = df['close'].copy()
    high = df['high'].copy()
    low = df['low'].copy()
    
    atr = np.zeros(n)
    tr = np.zeros(n)
    tr[0] = high.iloc[0] - low.iloc[0]
    atr[0] = tr[0]
    for i in range(1, n):
        tr[i] = np.max([high.iloc[i] - low.iloc[i], 
                       np.abs(high.iloc[i] - close.iloc[i-1]),
                       np.abs(low.iloc[i] - close.iloc[i-1])])
        atr[i] = (atr[i-1] * 9 + tr[i]) / 10
    
    threshold_multiplier = 3.0
    depth = 10
    deviation = atr / close * 100 * threshold_multiplier
    
    zigzag_high = np.full(n, np.nan)
    zigzag_low = np.full(n, np.nan)
    zigzag_high_idx = np.full(n, -1, dtype=int)
    zigzag_low_idx = np.full(n, -1, dtype=int)
    
    for i in range(depth, n):
        window_high = high.iloc[i-depth:i+1].max()
        window_low = low.iloc[i-depth:i+1].min()
        
        if high.iloc[i] >= window_high:
            is_pivot_high = True
            for j in range(max(0, i-depth), i):
                if high.iloc[j] > window_high:
                    is_pivot_high = False
                    break
            if is_pivot_high and high.iloc[i] >= window_high:
                zigzag_high[i] = high.iloc[i]
                zigzag_high_idx[i] = i
                
        if low.iloc[i] <= window_low:
            is_pivot_low = True
            for j in range(max(0, i-depth), i):
                if low.iloc[j] < window_low:
                    is_pivot_low = False
                    break
            if is_pivot_low and low.iloc[i] <= window_low:
                zigzag_low[i] = low.iloc[i]
                zigzag_low_idx[i] = i
    
    last_pivot_high = np.full(n, np.nan)
    last_pivot_low = np.full(n, np.nan)
    last_ph_idx = -1
    last_pl_idx = -1
    
    for i in range(n):
        if zigzag_high_idx[i] != -1:
            last_ph_idx = i
            last_pivot_high[i] = zigzag_high.iloc[i]
        if zigzag_low_idx[i] != -1:
            last_pl_idx = i
            last_pivot_low[i] = zigzag_low.iloc[i]
    
    reverse = False
    fib_0382 = 0.382
    fib_0786 = 0.786
    
    start_price = np.full(n, np.nan)
    end_price = np.full(n, np.nan)
    height = np.full(n, np.nan)
    r_0382 = np.full(n, np.nan)
    r_0786 = np.full(n, np.nan)
    
    for i in range(n):
        if last_ph_idx >= 0 and last_pl_idx >= 0:
            if last_ph_idx > last_pl_idx:
                sp = low.iloc[last_pl_idx]
                ep = high.iloc[last_ph_idx]
            else:
                sp = high.iloc[last_ph_idx]
                ep = low.iloc[last_pl_idx]
            
            if not reverse:
                start_price[i] = ep
                end_price[i] = sp
            else:
                start_price[i] = sp
                end_price[i] = ep
            
            h = (start_price[i] - end_price[i])
            if start_price[i] > end_price[i]:
                h = -1 * np.abs(start_price[i] - end_price[i])
            else:
                h = np.abs(start_price[i] - end_price[i])
            height[i] = h
            
            if not np.isnan(start_price[i]) and not np.isnan(height[i]):
                r_0382[i] = start_price[i] + height[i] * fib_0382
                r_0786[i] = start_price[i] + height[i] * fib_0786
    
    prev_close = close.shift(1)
    prev_r_0382 = r_0382.shift(1)
    prev_r_0786 = r_0786.shift(1)
    
    crossed_0382 = ((r_0382 > close) & (r_0382 < prev_close)) | ((r_0382 < close) & (r_0382 > prev_close))
    crossed_0786 = ((r_0786 > close) & (r_0786 < prev_close)) | ((r_0786 < close) & (r_0786 > prev_close))
    
    valid_zigzag = ~np.isnan(start_price) & ~np.isnan(height) & ~np.isnan(r_0382)
    
    adjusted_hour = np.zeros(n, dtype=int)
    adjusted_minute = np.zeros(n, dtype=int)
    for i in range(n):
        dt = datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc)
        adjusted_hour[i] = dt.hour
        adjusted_minute[i] = dt.minute
    
    start_hour = 7
    end_hour = 10
    end_minute = 59
    
    in_window = ((adjusted_hour >= start_hour) & (adjusted_hour <= end_hour) & 
                 ~((adjusted_hour == end_hour) & (adjusted_minute > end_minute)))
    
    bullish = height > 0
    
    long_condition = valid_zigzag & in_window & crossed_0382 & bullish
    
    for i in range(1, n):
        if long_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
    
    return entries