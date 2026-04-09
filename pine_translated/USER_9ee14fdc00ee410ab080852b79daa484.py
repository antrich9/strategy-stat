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
    trade_num = 0
    
    n = len(df)
    if n < 10:
        return entries
    
    # Time window detection (London session: 07:45-09:45 and 14:45-16:45)
    isWithinTimeWindow = np.zeros(n, dtype=bool)
    
    for i in range(n):
        dt = datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        time_minutes = hour * 60 + minute
        
        morning_start = 7 * 60 + 45   # 07:45
        morning_end = 9 * 60 + 45     # 09:45
        afternoon_start = 14 * 60 + 45  # 14:45
        afternoon_end = 16 * 60 + 45    # 16:45
        
        is_morning = morning_start <= time_minutes < morning_end
        is_afternoon = afternoon_start <= time_minutes < afternoon_end
        isWithinTimeWindow[i] = is_morning or is_afternoon
    
    # Get 4H data by resampling
    df['time_dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df_4h = df.set_index('time_dt').resample('240min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()
    
    # Previous Day High/Low (using 4H data lookback)
    prevDayHigh = np.full(n, np.nan)
    prevDayLow = np.full(n, np.nan)
    
    for i in range(n):
        current_time = df['time'].iloc[i]
        for j in range(len(df_4h)):
            if df_4h['time'].iloc[j] > current_time:
                if j > 0:
                    prevDayHigh[i] = df_4h['high'].iloc[j-1]
                    prevDayLow[i] = df_4h['low'].iloc[j-1]
                break
    
    # 4H data aligned to 15m bars
    high_4h = np.full(n, np.nan)
    low_4h = np.full(n, np.nan)
    
    df_indexed = df.set_index('time_dt')
    df_4h_indexed = df_4h.set_index('time')
    
    for i in range(n):
        current_time = df['time'].iloc[i]
        for j in range(len(df_4h)):
            if df_4h['time'].iloc[j] >= current_time:
                if j >= 2:
                    high_4h[i] = df_4h['high'].iloc[j-1]
                    low_4h[i] = df_4h['low'].iloc[j-1]
                break
    
    # Swing detection
    is_swing_high = np.zeros(n, dtype=bool)
    is_swing_low = np.zeros(n, dtype=bool)
    
    for i in range(5, n):
        h2 = high_4h[i]
        h1 = high_4h[i-1] if i-1 >= 0 else np.nan
        h3 = high_4h[i-2] if i-2 >= 0 else np.nan
        h4 = high_4h[i-3] if i-3 >= 0 else np.nan
        h5 = high_4h[i-4] if i-4 >= 0 else np.nan
        
        l2 = low_4h[i]
        l1 = low_4h[i-1] if i-1 >= 0 else np.nan
        l3 = low_4h[i-2] if i-2 >= 0 else np.nan
        l4 = low_4h[i-3] if i-3 >= 0 else np.nan
        l5 = low_4h[i-4] if i-4 >= 0 else np.nan
        
        if not (np.isnan(h5) or np.isnan(h4) or np.isnan(h3) or np.isnan(h2) or np.isnan(h1)):
            if h4 < h3 and h2 <= h3 and h3 >= h1 and h3 >= h5:
                is_swing_high[i] = True
        
        if not (np.isnan(l5) or np.isnan(l4) or np.isnan(l3) or np.isnan(l2) or np.isnan(l1)):
            if l4 > l3 and l2 >= l3 and l3 <= l1 and l3 <= l5:
                is_swing_low[i] = True
    
    # Track PDH/PDL sweeps
    flagpdh = False
    flagpdl = False
    previousDayHighTaken = np.zeros(n, dtype=bool)
    previousDayLowTaken = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        if not np.isnan(prevDayHigh[i]) and not np.isnan(prevDayLow[i]):
            if df['high'].iloc[i] > prevDayHigh[i]:
                previousDayHighTaken[i] = True
            if df['low'].iloc[i] < prevDayLow[i]:
                previousDayLowTaken[i] = True
    
    # Entry signals
    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        if isWithinTimeWindow[i]:
            if previousDayHighTaken[i] and not previousDayLowTaken[i]:
                long_entry[i] = True
            elif previousDayLowTaken[i] and not previousDayHighTaken[i]:
                short_entry[i] = True
    
    # Generate entries
    for i in range(1, n):
        if long_entry[i] or short_entry[i]:
            trade_num += 1
            direction = 'long' if long_entry[i] else 'short'
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
    
    return entries