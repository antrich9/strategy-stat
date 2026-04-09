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
    
    close = df['close']
    
    # EMAs for current timeframe (cema9 and cema18)
    cema9 = close.ewm(span=9, adjust=False).mean()
    cema18 = close.ewm(span=18, adjust=False).mean()
    
    # Resample to 4H for ema9 and ema18
    df_temp = df.copy()
    df_temp['ts_4h'] = (df_temp['time'] // (4 * 3600 * 1000)) * (4 * 3600 * 1000)
    resampled_4h = df_temp.groupby('ts_4h')['close'].last()
    
    ema9_4h_series = resampled_4h.ewm(span=9, adjust=False).mean()
    ema18_4h_series = resampled_4h.ewm(span=18, adjust=False).mean()
    
    df_temp['ema9_4h'] = df_temp['ts_4h'].map(ema9_4h_series.to_dict())
    df_temp['ema18_4h'] = df_temp['ts_4h'].map(ema18_4h_series.to_dict())
    
    ema9_4h = df_temp['ema9_4h']
    ema18_4h = df_temp['ema18_4h']
    
    # Get previous day high and low
    df_temp['day_ts'] = df_temp['time'] // (24 * 3600 * 1000)
    day_high = df_temp.groupby('day_ts')['high'].max().shift(1)
    day_low = df_temp.groupby('day_ts')['low'].min().shift(1)
    df_temp['prev_day_high'] = df_temp['day_ts'].map(day_high.to_dict())
    df_temp['prev_day_low'] = df_temp['day_ts'].map(day_low.to_dict())
    
    # Time filtering for 07:00-09:59 and 12:00-14:59 UTC
    def get_hour(ts):
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).hour
    hours = df_temp['time'].apply(get_hour)
    time_filter_1 = (hours >= 7) & (hours < 10)
    time_filter_2 = (hours >= 12) & (hours < 15)
    time_filter = time_filter_1 | time_filter_2
    
    # Entry conditions
    condition_long = (close > ema9_4h) & (close > ema18_4h) & (close > cema9) & (close > cema18)
    condition_short = (close < ema9_4h) & (close < ema18_4h) & (close < cema9) & (close < cema18)
    
    # Flags for liquidity sweeps
    prev_day_high_swept = close > df_temp['prev_day_high']
    prev_day_low_swept = close < df_temp['prev_day_low']
    
    flagpdh = prev_day_high_swept.copy()
    flagpdl = prev_day_low_swept.copy()
    waitingForEntry = False
    waitingForShortEntry1 = False
    
    entries = []
    trade_num = 1
    
    for i in range(1, len(df_temp)):
        if pd.isna(ema9_4h.iloc[i]) or pd.isna(ema18_4h.iloc[i]) or pd.isna(cema9.iloc[i]) or pd.isna(cema18.iloc[i]):
            continue
        
        if pd.isna(df_temp['prev_day_high'].iloc[i]) or pd.isna(df_temp['prev_day_low'].iloc[i]):
            continue
        
        # New day reset
        if i > 0 and df_temp['day_ts'].iloc[i] != df_temp['day_ts'].iloc[i-1]:
            flagpdh.iloc[i] = False
            flagpdl.iloc[i] = False
            waitingForEntry = False
            waitingForShortEntry1 = False
        
        # Check for sweeps
        if close.iloc[i] > df_temp['prev_day_high'].iloc[i]:
            flagpdh.iloc[i] = True
        if close.iloc[i] < df_temp['prev_day_low'].iloc[i]:
            flagpdl.iloc[i] = True
        
        # Reset flags on new day
        if i > 0 and df_temp['day_ts'].iloc[i] != df_temp['day_ts'].iloc[i-1]:
            flagpdh.iloc[i] = False
            flagpdl.iloc[i] = False
            waitingForEntry = False
            waitingForShortEntry1 = False
            if i > 0:
                flagpdh.iloc[i] = flagpdh.iloc[i-1]
                flagpdl.iloc[i] = flagpdl.iloc[i-1]
        
        # Long entry logic
        if not waitingForEntry and flagpdl.iloc[i] and condition_long.iloc[i] and time_filter.iloc[i]:
            waitingForEntry = True
        
        if waitingForEntry and condition_long.iloc[i]:
            ts = int(df_temp['time'].iloc[i])
            entry = {
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
            }
            entries.append(entry)
            trade_num += 1
            waitingForEntry = False
        
        # Short entry logic
        if not waitingForShortEntry1 and flagpdh.iloc[i] and condition_short.iloc[i] and time_filter.iloc[i]:
            waitingForShortEntry1 = True
        
        if waitingForShortEntry1 and condition_short.iloc[i]:
            ts = int(df_temp['time'].iloc[i])
            entry = {
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            }
            entries.append(entry)
            trade_num += 1
            waitingForShortEntry1 = False
    
    return entries