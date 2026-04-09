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
    
    # Convert time to datetime for session checks
    df['_dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Session parameters (using Pine defaults)
    asia_start_hour = 20
    asia_start_minute = 0
    asia_end_hour = 0
    asia_end_minute = 0
    timezone_asia = 'America/New_York'
    
    london_start_hour = 3
    london_start_minute = 0
    london_end_hour = 7
    london_end_minute = 0
    timezone_london = 'America/New_York'
    
    # Filter flags (using Pine defaults - all False)
    inp1 = False  # Volume Filter
    inp2 = False  # ATR Filter
    inp3 = False  # Trend Filter
    
    # Asia session timestamps
    asia_start_ts_list = []
    asia_end_ts_list = []
    for i in range(len(df)):
        dt = df['_dt'].iloc[i]
        day_offset_end = 1 if asia_end_hour <= asia_start_hour else 0
        
        start_dt = dt.replace(hour=asia_start_hour, minute=asia_start_minute, second=0, microsecond=0)
        end_dt = dt.replace(hour=asia_end_hour, minute=asia_end_minute, second=0, microsecond=0)
        if day_offset_end == 1:
            end_dt += pd.Timedelta(days=1)
        
        asia_start_ts_list.append(int(start_dt.timestamp()))
        asia_end_ts_list.append(int(end_dt.timestamp()))
    
    df['asia_session_start'] = asia_start_ts_list
    df['asia_session_end'] = asia_end_ts_list
    
    # London session timestamps
    london_start_ts_list = []
    london_end_ts_list = []
    for i in range(len(df)):
        dt = df['_dt'].iloc[i]
        day_offset_end = 1 if london_end_hour <= london_start_hour else 0
        
        start_dt = dt.replace(hour=london_start_hour, minute=london_start_minute, second=0, microsecond=0)
        end_dt = dt.replace(hour=london_end_hour, minute=london_end_minute, second=0, microsecond=0)
        if day_offset_end == 1:
            end_dt += pd.Timedelta(days=1)
        
        london_start_ts_list.append(int(start_dt.timestamp()))
        london_end_ts_list.append(int(end_dt.timestamp()))
    
    df['london_session_start'] = london_start_ts_list
    df['london_session_end'] = london_end_ts_list
    
    # Session flags
    df['in_asia_session'] = (df['time'] >= df['asia_session_start']) & (df['time'] < df['asia_session_end'])
    df['in_london_session'] = (df['time'] >= df['london_session_start']) & (df['time'] < df['london_session_end'])
    
    # Asia session high/low tracking
    asia_high = pd.Series([np.nan] * len(df), index=df.index)
    asia_low = pd.Series([np.nan] * len(df), index=df.index)
    asia_swept_high = pd.Series([False] * len(df), index=df.index)
    asia_swept_low = pd.Series([False] * len(df), index=df.index)
    asia_sweep_high_now = pd.Series([False] * len(df), index=df.index)
    asia_sweep_low_now = pd.Series([False] * len(df), index=df.index)
    
    for i in range(len(df)):
        if i == 0:
            continue
        
        # Session start detection
        prev_time_in_asia = df['time'].iloc[i-1] < df['asia_session_start'].iloc[i] and df['time'].iloc[i] >= df['asia_session_start'].iloc[i]
        
        if prev_time_in_asia:
            asia_high.iloc[i] = np.nan
            asia_low.iloc[i] = np.nan
            asia_swept_high.iloc[i] = False
            asia_swept_low.iloc[i] = False
        else:
            asia_high.iloc[i] = asia_high.iloc[i-1]
            asia_low.iloc[i] = asia_low.iloc[i-1]
            asia_swept_high.iloc[i] = asia_swept_high.iloc[i-1]
            asia_swept_low.iloc[i] = asia_swept_low.iloc[i-1]
        
        if df['in_asia_session'].iloc[i]:
            if np.isnan(asia_high.iloc[i]) or pd.isna(asia_high.iloc[i]):
                asia_high.iloc[i] = df['high'].iloc[i]
            else:
                asia_high.iloc[i] = max(asia_high.iloc[i-1], df['high'].iloc[i])
            
            if np.isnan(asia_low.iloc[i]) or pd.isna(asia_low.iloc[i]):
                asia_low.iloc[i] = df['low'].iloc[i]
            else:
                asia_low.iloc[i] = min(asia_low.iloc[i-1], df['low'].iloc[i])
        
        # Sweep detection
        if not df['in_asia_session'].iloc[i] and not asia_swept_high.iloc[i] and not pd.isna(asia_high.iloc[i]) and df['high'].iloc[i] > asia_high.iloc[i]:
            asia_sweep_high_now.iloc[i] = True
            asia_swept_high.iloc[i] = True
        elif not df['in_asia_session'].iloc[i] and not asia_swept_low.iloc[i] and not pd.isna(asia_low.iloc[i]) and df['low'].iloc[i] < asia_low.iloc[i]:
            asia_sweep_low_now.iloc[i] = True
            asia_swept_low.iloc[i] = True
    
    # London session high/low tracking
    london_high = pd.Series([np.nan] * len(df), index=df.index)
    london_low = pd.Series([np.nan] * len(df), index=df.index)
    london_swept_high = pd.Series([False] * len(df), index=df.index)
    london_swept_low = pd.Series([False] * len(df), index=df.index)
    london_sweep_high_now = pd.Series([False] * len(df), index=df.index)
    london_sweep_low_now = pd.Series([False] * len(df), index=df.index)
    
    for i in range(len(df)):
        if i == 0:
            continue
        
        prev_time_in_london = df['time'].iloc[i-1] < df['london_session_start'].iloc[i] and df['time'].iloc[i] >= df['london_session_start'].iloc[i]
        
        if prev_time_in_london:
            london_high.iloc[i] = np.nan
            london_low.iloc[i] = np.nan
            london_swept_high.iloc[i] = False
            london_swept_low.iloc[i] = False
        else:
            london_high.iloc[i] = london_high.iloc[i-1]
            london_low.iloc[i] = london_low.iloc[i-1]
            london_swept_high.iloc[i] = london_swept_high.iloc[i-1]
            london_swept_low.iloc[i] = london_swept_low.iloc[i-1]
        
        if df['in_london_session'].iloc[i]:
            if np.isnan(london_high.iloc[i]) or pd.isna(london_high.iloc[i]):
                london_high.iloc[i] = df['high'].iloc[i]
            else:
                london_high.iloc[i] = max(london_high.iloc[i-1], df['high'].iloc[i])
            
            if np.isnan(london_low.iloc[i]) or pd.isna(london_low.iloc[i]):
                london_low.iloc[i] = df['low'].iloc[i]
            else:
                london_low.iloc[i] = min(london_low.iloc[i-1], df['low'].iloc[i])
        
        if not df['in_london_session'].iloc[i] and not london_swept_high.iloc[i] and not pd.isna(london_high.iloc[i]) and df['high'].iloc[i] > london_high.iloc[i]:
            london_sweep_high_now.iloc[i] = True
            london_swept_high.iloc[i] = True
        elif not df['in_london_session'].iloc[i] and not london_swept_low.iloc[i] and not pd.isna(london_low.iloc[i]) and df['low'].iloc[i] < london_low.iloc[i]:
            london_sweep_low_now.iloc[i] = True
            london_swept_low.iloc[i] = True
    
    # FVG and filters
    df['high_2'] = df['high'].shift(2)
    df['low_2'] = df['low'].shift(2)
    
    volfilt = (~inp1) | (df['volume'].shift(1) > df['volume'].rolling(9).mean() * 1.5)
    
    # Wilder ATR
    high_low = df['high'] - df['low']
    high_low_prev = high_low.shift(1)
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/20, adjust=False).mean()
    atr_filter_val = atr / 1.5
    atrfilt = (~inp2) | ((df['low'] - df['high_2'] > atr_filter_val) | (df['low_2'] - df['high'] > atr_filter_val))
    
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = (~inp3) | loc2
    locfilts = (~inp3) | (~loc2)
    
    df['bfvg'] = (df['low'] > df['high_2']) & volfilt & atrfilt & locfiltb
    df['sfvg'] = (df['high'] < df['low_2']) & volfilt & atrfilt & locfilts
    
    # Build entry conditions
    long_condition = df['bfvg']
    short_condition = df['sfvg']
    
    # Generate entries
    for i in range(len(df)):
        if pd.isna(df['low'].iloc[i]) or pd.isna(df['high_2'].iloc[i]) or pd.isna(loc.iloc[i]):
            continue
        
        if long_condition.iloc[i]:
            entry_price = df['close'].iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
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
        
        if short_condition.iloc[i]:
            entry_price = df['close'].iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
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
    
    return results