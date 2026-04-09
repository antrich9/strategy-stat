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
    ts = pd.to_datetime(df['time'], unit='ms', utc=True)
    
    def in_london_window(hour, minute):
        in_morning = (hour == 8) or (hour == 9 and minute <= 45)
        in_afternoon = (hour == 16) or (hour == 16 and minute <= 45)
        return in_morning or in_afternoon
    
    hours = ts.dt.hour
    minutes = ts.dt.minute
    is_within_time_window = df.index >= 0
    for i in df.index:
        h = hours.iloc[i] if i < len(hours) else 0
        m = minutes.iloc[i] if i < len(minutes) else 0
        is_within_time_window.iloc[i] = in_london_window(h, m)
    
    current_day = ts.dt.date
    is_new_day = (current_day != current_day.shift(1).fillna(current_day))
    
    prev_day_h = df['high'].shift(1).copy()
    prev_day_l = df['low'].shift(1).copy()
    
    for i in range(1, len(df)):
        if is_new_day.iloc[i]:
            day_high = df['high'].iloc[:i].max() if i > 0 else np.nan
            day_low = df['low'].iloc[:i].min() if i > 0 else np.nan
            prev_day_h.iloc[i] = day_high
            prev_day_l.iloc[i] = day_low
        else:
            prev_day_h.iloc[i] = prev_day_h.iloc[i-1]
            prev_day_l.iloc[i] = prev_day_l.iloc[i-1]
    
    previous_day_high_taken = (df['high'] > prev_day_h) & (df['high'].shift(1) <= prev_day_h)
    previous_day_low_taken = (df['low'] < prev_day_l) & (df['low'].shift(1) >= prev_day_l)
    previous_day_high_taken.iloc[0] = False
    previous_day_low_taken.iloc[0] = False
    
    flagpdh = pd.Series(False, index=df.index)
    flagpdl = pd.Series(False, index=df.index)
    
    for i in range(1, len(df)):
        if is_new_day.iloc[i]:
            flagpdh.iloc[i] = False
            flagpdl.iloc[i] = False
        else:
            if previous_day_high_taken.iloc[i] and df['low'].iloc[i] > prev_day_l.iloc[i]:
                flagpdh.iloc[i] = True
            if previous_day_low_taken.iloc[i] and df['high'].iloc[i] < prev_day_h.iloc[i]:
                flagpdl.iloc[i] = True
    
    entries = []
    trade_num = 1
    
    taken_long_today = False
    taken_short_today = False
    prev_date = None
    
    for i in range(1, len(df)):
        cur_date = current_day.iloc[i]
        if cur_date != prev_date:
            taken_long_today = False
            taken_short_today = False
            prev_date = cur_date
        
        if pd.isna(df['high'].iloc[i]) or pd.isna(df['low'].iloc[i]) or pd.isna(prev_day_h.iloc[i]) or pd.isna(prev_day_l.iloc[i]):
            continue
        
        if not is_within_time_window.iloc[i]:
            continue
        
        if flagpdl.iloc[i] and not taken_long_today:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
            taken_long_today = True
        
        if flagpdh.iloc[i] and not taken_short_today:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
            taken_short_today = True
    
    entries.sort(key=lambda x: x['entry_ts'])
    for idx, entry in enumerate(entries, start=1):
        entry['trade_num'] = idx
    
    return entries