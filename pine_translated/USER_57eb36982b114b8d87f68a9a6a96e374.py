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
    fastLength = 50
    slowLength = 200
    
    close = df['close']
    high = df['high']
    low = df['low']
    open_price = df['open']
    
    fastEMA = close.ewm(span=fastLength, adjust=False).mean()
    slowEMA = close.ewm(span=slowLength, adjust=False).mean()
    
    crossOver = (fastEMA > slowEMA) & (fastEMA.shift(1) <= slowEMA.shift(1))
    crossUnder = (fastEMA < slowEMA) & (fastEMA.shift(1) >= slowEMA.shift(1))
    
    df['fastEMA'] = fastEMA
    df['slowEMA'] = slowEMA
    df['crossOver'] = crossOver
    df['crossUnder'] = crossUnder
    
    timestamps = df['time']
    
    def is_time_in_range(ts, start_h, start_m, end_h, end_m, gmt_offset=1):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        local_hour = (dt.hour + gmt_offset) % 24
        local_minute = dt.minute
        current_total_minutes = local_hour * 60 + local_minute
        start_total_minutes = start_h * 60 + start_m
        end_total_minutes = end_h * 60 + end_m
        return start_total_minutes <= current_total_minutes <= end_total_minutes
    
    morning_filter = timestamps.apply(lambda x: is_time_in_range(x, 7, 0, 9, 59, gmt_offset=1))
    afternoon_filter = timestamps.apply(lambda x: is_time_in_range(x, 12, 0, 14, 59, gmt_offset=1))
    time_filter = morning_filter | afternoon_filter
    
    df['time_filter'] = time_filter
    
    def get_prev_day_high_low(idx, lookback=5):
        ts = df['time'].iloc[idx]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        day_start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        day_start_ts = int(day_start.timestamp())
        
        prev_day_mask = df['time'] < day_start_ts
        if not prev_day_mask.any():
            return np.nan, np.nan
        
        prev_day_data = df[prev_day_mask]
        if len(prev_day_data) == 0:
            return np.nan, np.nan
        
        prev_day_high_val = prev_day_data['high'].max()
        prev_day_low_val = prev_day_data['low'].min()
        
        return prev_day_high_val, prev_day_low_val
    
    flagpdl = pd.Series(False, index=df.index)
    flagpdh = pd.Series(False, index=df.index)
    
    prev_day_high = pd.Series(np.nan, index=df.index)
    prev_day_low = pd.Series(np.nan, index=df.index)
    
    for i in range(1, len(df)):
        dt = datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc)
        prev_dt = datetime.fromtimestamp(df['time'].iloc[i-1], tz=timezone.utc)
        
        if dt.date() != prev_dt.date():
            flagpdl.iloc[i] = False
            flagpdh.iloc[i] = False
        else:
            flagpdl.iloc[i] = flagpdl.iloc[i-1]
            flagpdh.iloc[i] = flagpdh.iloc[i-1]
        
        pdh, pdl = get_prev_day_high_low(i)
        prev_day_high.iloc[i] = pdh
        prev_day_low.iloc[i] = pdl
        
        if not np.isnan(pdh) and df['close'].iloc[i] > pdh:
            flagpdh.iloc[i] = True
        if not np.isnan(pdl) and df['close'].iloc[i] < pdl:
            flagpdl.iloc[i] = True
    
    df['flagpdl'] = flagpdl
    df['flagpdh'] = flagpdh
    df['prev_day_high'] = prev_day_high
    df['prev_day_low'] = prev_day_low
    
    def is_up(idx):
        if idx >= len(df):
            return False
        return df['close'].iloc[idx] > df['open'].iloc[idx]
    
    def is_down(idx):
        if idx >= len(df):
            return False
        return df['close'].iloc[idx] < df['open'].iloc[idx]
    
    def is_ob_up(idx):
        if idx + 1 >= len(df) or idx >= len(df):
            return False
        return is_down(idx + 1) and is_up(idx) and df['close'].iloc[idx] > df['high'].iloc[idx + 1]
    
    def is_ob_down(idx):
        if idx + 1 >= len(df) or idx >= len(df):
            return False
        return is_up(idx + 1) and is_down(idx) and df['close'].iloc[idx] < df['low'].iloc[idx + 1]
    
    def is_fvg_up(idx):
        if idx + 2 >= len(df) or idx >= len(df):
            return False
        return df['low'].iloc[idx] > df['high'].iloc[idx + 2]
    
    def is_fvg_down(idx):
        if idx + 2 >= len(df) or idx >= len(df):
            return False
        return df['high'].iloc[idx] < df['low'].iloc[idx + 2]
    
    bull_ob = pd.Series(False, index=df.index)
    bear_ob = pd.Series(False, index=df.index)
    bull_fvg = pd.Series(False, index=df.index)
    bear_fvg = pd.Series(False, index=df.index)
    
    for i in range(2, len(df)):
        bull_ob.iloc[i] = is_ob_up(i - 1)
        bear_ob.iloc[i] = is_ob_down(i - 1)
        bull_fvg.iloc[i] = is_fvg_up(i - 2)
        bear_fvg.iloc[i] = is_fvg_down(i - 2)
    
    df['bull_ob'] = bull_ob
    df['bear_ob'] = bear_ob
    df['bull_fvg'] = bull_fvg
    df['bear_fvg'] = bear_fvg
    
    bull_stacked = bull_ob & bull_fvg
    bear_stacked = bear_ob & bear_fvg
    
    df['bull_stacked'] = bull_stacked
    df['bear_stacked'] = bear_stacked
    
    long_condition = (df['crossOver']) & (df['flagpdl']) & (df['time_filter']) & (df['bull_stacked'])
    short_condition = (df['crossUnder']) & (df['flagpdh']) & (df['time_filter']) & (df['bear_stacked'])
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(df['fastEMA'].iloc[i]) or pd.isna(df['slowEMA'].iloc[i]):
            continue
        
        if long_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            
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
        
        elif short_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            
            entries.append({
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
    
    return entries