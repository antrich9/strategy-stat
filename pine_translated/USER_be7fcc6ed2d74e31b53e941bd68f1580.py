import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.sort_values('time').reset_index(drop=True)
    
    # Calculate previous day high/low using daily resampling
    df_temp = df.copy()
    df_temp['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_localize(None)
    df_temp.set_index('datetime', inplace=True)
    
    daily_ohlc = df_temp.resample('D').agg({
        'high': 'max',
        'low': 'min'
    }).dropna()
    
    prev_day_high = daily_ohlc['high'].shift(1).reindex(df_temp.index, method='ffill')
    prev_day_low = daily_ohlc['low'].shift(1).reindex(df_temp.index, method='ffill')
    
    # Check if in London session (7:45-9:45 and 15:45-16:45 Europe/London)
    dt = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert('Europe/London')
    hour = dt.dt.hour
    minute = dt.dt.minute
    time_in_mins = hour * 60 + minute
    in_london_session = ((time_in_mins >= 7 * 60 + 45) & (time_in_mins < 9 * 60 + 45)) | \
                        ((time_in_mins >= 15 * 60 + 45) & (time_in_mins < 16 * 60 + 45))
    
    # isFriday: dayofweek == dayofweek.friday (Friday = 4 in Python weekday)
    is_friday = dt.dt.weekday == 4
    
    # BullFVG = barstate.isconfirmed and low[1] > high[3]
    bullFVG = df['low'].shift(1) > df['high'].shift(3)
    
    # BearFVG = barstate.isconfirmed and high[1] < low[3]
    bearFVG = df['high'].shift(1) < df['low'].shift(3)
    
    # flagPDL = ta.cross(low, prev_day_low)
    flagPDL = ((df['low'] >= prev_day_low) & (df['low'].shift(1) < prev_day_low)) | \
              ((df['low'] <= prev_day_low) & (df['low'].shift(1) > prev_day_low))
    
    # flagPDH = ta.cross(high, prev_day_high)
    flagPDH = ((df['high'] >= prev_day_high) & (df['high'].shift(1) < prev_day_high)) | \
              ((df['high'] <= prev_day_high) & (df['high'].shift(1) > prev_day_high))
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if i < 4:
            continue
        if pd.isna(prev_day_high.iloc[i]) or pd.isna(prev_day_low.iloc[i]):
            continue
        if pd.isna(bullFVG.iloc[i]) or pd.isna(bearFVG.iloc[i]):
            continue
        
        bar_time = df['time'].iloc[i]
        bar_dt = datetime.fromtimestamp(bar_time, tz=timezone.utc)
        bar_dt_london = bar_dt.astimezone(timezone.utc).replace(tzinfo=None)
        try:
            from zoneinfo import ZoneInfo
            bar_dt_london = bar_dt.astimezone(ZoneInfo('Europe/London')).replace(tzinfo=None)
        except:
            pass
        hour = bar_dt_london.hour
        minute = bar_dt_london.minute
        time_in_mins = hour * 60 + minute
        in_london = ((7 * 60 + 45 <= time_in_mins < 9 * 60 + 45) or 
                     (15 * 60 + 45 <= time_in_mins < 16 * 60 + 45))
        if not in_london:
            continue
        try:
            from zoneinfo import ZoneInfo
            bar_dt_check = bar_dt.astimezone(ZoneInfo('Europe/London'))
        except:
            bar_dt_check = bar_dt
        is_fri = (bar_dt_check.weekday() == 4)
        if is_fri:
            continue
        
        if flagPDL.iloc[i] and bullFVG.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': bar_time,
                'entry_time': bar_dt.isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        
        if flagPDH.iloc[i] and bearFVG.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': bar_time,
                'entry_time': bar_dt.isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries