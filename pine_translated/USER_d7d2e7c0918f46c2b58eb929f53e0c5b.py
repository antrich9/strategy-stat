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
    
    # Create copies to avoid SettingWithCopyWarning
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Resample to daily to get previous day high/low
    daily = df.set_index('time').resample('D').agg({'high': 'max', 'low': 'min', 'open': 'first', 'close': 'last'})
    daily = daily.dropna()
    daily['prev_day_high'] = daily['high'].shift(1)
    daily['prev_day_low'] = daily['low'].shift(1)
    daily = daily.dropna()
    
    # Merge previous day high/low back to intrabar data
    df['day'] = df['time'].dt.date
    daily = daily.reset_index()
    daily['day'] = daily['time'].dt.date
    daily = daily[['day', 'prev_day_high', 'prev_day_low']]
    df = df.merge(daily, on='day', how='left')
    df['prev_day_high'] = df['prev_day_high'].ffill()
    df['prev_day_low'] = df['prev_day_low'].ffill()
    df = df.dropna(subset=['prev_day_high', 'prev_day_low'])
    
    # Detect sweeps of previous day high/low
    df['sweep_pdh'] = (df['close'] > df['prev_day_high']) & (df['prev_day_high'].shift(1) < df['prev_day_high'].shift(1).shift(1))
    df['sweep_pdl'] = (df['close'] < df['prev_day_low']) & (df['prev_day_low'].shift(1) > df['prev_day_low'].shift(1).shift(1))
    
    # OB detection functions
    def is_up(idx):
        return df['close'].iloc[idx] > df['open'].iloc[idx]
    
    def is_down(idx):
        return df['close'].iloc[idx] < df['open'].iloc[idx]
    
    def is_ob_up(idx):
        return is_down(idx + 1) and is_up(idx) and df['close'].iloc[idx] > df['high'].iloc[idx + 1]
    
    def is_ob_down(idx):
        return is_up(idx + 1) and is_down(idx) and df['close'].iloc[idx] < df['low'].iloc[idx + 1]
    
    def is_fvg_up(idx):
        return df['low'].iloc[idx] > df['high'].iloc[idx + 2]
    
    def is_fvg_down(idx):
        return df['high'].iloc[idx] < df['low'].iloc[idx + 2]
    
    # Build boolean series for OB and FVG conditions
    df['ob_up'] = False
    df['ob_down'] = False
    df['fvg_up'] = False
    df['fvg_down'] = False
    
    for i in range(3, len(df) - 2):
        if pd.notna(df['close'].iloc[i+1]) and pd.notna(df['open'].iloc[i+1]):
            df.iloc[i, df.columns.get_loc('ob_up')] = is_ob_up(i)
            df.iloc[i, df.columns.get_loc('ob_down')] = is_ob_down(i)
        if pd.notna(df['low'].iloc[i]) and pd.notna(df['high'].iloc[i+2]):
            df.iloc[i, df.columns.get_loc('fvg_up')] = is_fvg_up(i)
        if pd.notna(df['high'].iloc[i]) and pd.notna(df['low'].iloc[i+2]):
            df.iloc[i, df.columns.get_loc('fvg_down')] = is_fvg_down(i)
    
    # Time filter functions (GMT+1 timezone, 07:00-09:59 and 12:00-14:59)
    def is_within_time_window(ts):
        dt = ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        time_mins = hour * 60 + minute
        return (700 <= time_mins <= 959) or (1200 <= time_mins <= 1459)
    
    df['in_time_window'] = df['time'].apply(is_within_time_window)
    
    # Initialize flags
    df['flag_pdh'] = False
    df['flag_pdl'] = False
    df['waiting_for_entry'] = False
    df['waiting_for_short_entry'] = False
    
    # Track previous day for reset
    df['day_change'] = df['day'] != df['day'].shift(1)
    
    prev_pdh_swept = False
    prev_pdl_swept = False
    waiting_for_entry = False
    waiting_for_short_entry = False
    
    for i in range(len(df)):
        if df.iloc[i]['day_change'] and i > 0:
            prev_pdh_swept = False
            prev_pdl_swept = False
            waiting_for_entry = False
            waiting_for_short_entry = False
        
        prev_day_high = df.iloc[i]['prev_day_high']
        prev_day_low = df.iloc[i]['prev_day_low']
        close_price = df.iloc[i]['close']
        
        if close_price > prev_day_high and not prev_pdh_swept:
            prev_pdh_swept = True
        if close_price < prev_day_low and not prev_pdl_swept:
            prev_pdl_swept = True
        
        df.iloc[i, df.columns.get_loc('flag_pdh')] = prev_pdh_swept
        df.iloc[i, df.columns.get_loc('flag_pdl')] = prev_pdl_swept
        df.iloc[i, df.columns.get_loc('waiting_for_entry')] = waiting_for_entry
        df.iloc[i, df.columns.get_loc('waiting_for_short_entry')] = waiting_for_short_entry
        
        in_time = df.iloc[i]['in_time_window']
        ob_up = df.iloc[i]['ob_up']
        fvg_up = df.iloc[i]['fvg_up']
        ob_down = df.iloc[i]['ob_down']
        fvg_down = df.iloc[i]['fvg_down']
        
        entry_time = df.iloc[i]['time']
        entry_ts = int(entry_time.timestamp())
        
        # Long entry condition
        if prev_pdh_swept and in_time and ob_up and fvg_up:
            entry_price = close_price
            entry_iso = entry_time.isoformat()
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_iso,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            waiting_for_entry = True
        
        # Short entry condition
        if prev_pdl_swept and in_time and ob_down and fvg_down:
            entry_price = close_price
            entry_iso = entry_time.isoformat()
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_iso,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            waiting_for_short_entry = True
    
    return results