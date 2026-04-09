import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    gmtSelect = "GMT+1"
    betweenTime_start = "0700"
    betweenTime_end = "0959"
    betweenTime1_start = "1200"
    betweenTime1_end = "1459"
    
    def is_time_in_window(ts, start_str, end_str, tz_str):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hours = int(start_str[:2])
        minutes = int(start_str[2:])
        start_hours = int(end_str[:2])
        start_minutes = int(end_str[2:])
        
        dt_utc = dt.replace(tzinfo=timezone.utc)
        
        if start_hours > hours or (start_hours == hours and start_minutes > minutes):
            start = dt_utc.replace(hour=hours, minute=minutes, second=0)
            end = dt_utc.replace(hour=start_hours, minute=start_minutes, second=0)
        else:
            start = dt_utc.replace(hour=hours, minute=minutes, second=0)
            end = (dt_utc.replace(hour=start_hours, minute=start_minutes, second=0))
            if end <= start:
                end = end + pd.Timedelta(days=1)
        
        return start <= dt_utc <= end
    
    df['time_dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    df['prevDayHigh'] = df['high'].shift(1).rolling(window=2, min_periods=2).max()
    df['prevDayLow'] = df['low'].shift(1).rolling(window=2, min_periods=2).min()
    
    df['isNewDay'] = df['time_dt'].diff().dt.days.fillna(0) > 0
    
    df['flagpdl'] = False
    df['flagpdh'] = False
    df['waitingForEntry'] = False
    df['waitingForShortEntry1'] = False
    
    flagpdl = False
    flagpdh = False
    waitingForEntry = False
    waitingForShortEntry1 = False
    
    for i in range(1, len(df)):
        if df['isNewDay'].iloc[i]:
            flagpdl = False
            flagpdh = False
            waitingForEntry = False
            waitingForShortEntry1 = False
        
        if df['close'].iloc[i] > df['prevDayHigh'].iloc[i]:
            flagpdh = True
        if df['close'].iloc[i] < df['prevDayLow'].iloc[i]:
            flagpdl = True
        
        df.iloc[i, df.columns.get_loc('flagpdl')] = flagpdl
        df.iloc[i, df.columns.get_loc('flagpdh')] = flagpdh
        df.iloc[i, df.columns.get_loc('waitingForEntry')] = waitingForEntry
        df.iloc[i, df.columns.get_loc('waitingForShortEntry1')] = waitingForShortEntry1
    
    df['isUp'] = df['close'] > df['open']
    df['isDown'] = df['close'] < df['open']
    
    df['obUp'] = df['isDown'].shift(1) & df['isUp'] & (df['close'] > df['high'].shift(1))
    df['obDown'] = df['isUp'].shift(1) & df['isDown'] & (df['close'] < df['low'].shift(1))
    df['fvgUp'] = df['low'] > df['high'].shift(2)
    df['fvgDown'] = df['high'] < df['low'].shift(2)
    
    df['inTimeWindow'] = df['time'].apply(lambda x: is_time_in_window(x, betweenTime_start, betweenTime_end, gmtSelect))
    df['inTimeWindow1'] = df['time'].apply(lambda x: is_time_in_window(x, betweenTime1_start, betweenTime1_end, gmtSelect))
    
    df['longCondition'] = df['flagpdl'] & df['inTimeWindow'] & df['obUp'] & df['fvgUp']
    df['shortCondition'] = df['flagpdh'] & df['inTimeWindow1'] & df['obDown'] & df['fvgDown']
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if i < 3:
            continue
        if pd.isna(df['obUp'].iloc[i]) or pd.isna(df['fvgUp'].iloc[i]) or pd.isna(df['obDown'].iloc[i]) or pd.isna(df['fvgDown'].iloc[i]):
            continue
        
        if df['longCondition'].iloc[i]:
            ts = int(df['time'].iloc[i])
            entry = {
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            }
            entries.append(entry)
            trade_num += 1
        
        if df['shortCondition'].iloc[i]:
            ts = int(df['time'].iloc[i])
            entry = {
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            }
            entries.append(entry)
            trade_num += 1
    
    return entries