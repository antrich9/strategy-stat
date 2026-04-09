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
    
    # Parameters from Pine Script
    fastLength = 50
    slowLength = 200
    gmtSelect = "GMT+1"
    
    # Time filter session strings
    betweenTime_start = "0700"
    betweenTime_end = "0959"
    betweenTime1_start = "1200"
    betweenTime1_end = "1459"
    
    # Calculate EMAs
    fastEMA = df['close'].ewm(span=fastLength, adjust=False).mean()
    slowEMA = df['close'].ewm(span=slowLength, adjust=False).mean()
    
    # Build boolean Series for conditions
    # Crossover: fast crosses above slow
    crossover_long = (fastEMA > slowEMA) & (fastEMA.shift(1) <= slowEMA.shift(1))
    
    # Crossunder: fast crosses below slow
    crossunder_short = (fastEMA < slowEMA) & (fastEMA.shift(1) >= slowEMA.shift(1))
    
    # Time filter conditions
    times = pd.to_datetime(df['time'], unit='s', utc=True)
    hours = times.dt.hour
    minutes = times.dt.minute
    total_minutes = hours * 60 + minutes
    
    # GMT+1 offset: subtract 1 hour to get GMT time
    gmt_time_offset = -1
    gmt_total_minutes = (total_minutes + gmt_time_offset * 60) % (24 * 60)
    
    betweenTime_start_min = int(betweenTime_start[:2]) * 60 + int(betweenTime_start[2:])
    betweenTime_end_min = int(betweenTime_end[:2]) * 60 + int(betweenTime_end[2:])
    betweenTime1_start_min = int(betweenTime1_start[:2]) * 60 + int(betweenTime1_start[2:])
    betweenTime1_end_min = int(betweenTime1_end[:2]) * 60 + int(betweenTime1_end[2:])
    
    time_filter_long = (gmt_total_minutes >= betweenTime_start_min) & (gmt_total_minutes <= betweenTime_end_min)
    time_filter_short = (gmt_total_minutes >= betweenTime1_start_min) & (gmt_total_minutes <= betweenTime1_end_min)
    
    # OB/FVG conditions
    close_vals = df['close'].values
    open_vals = df['open'].values
    high_vals = df['high'].values
    low_vals = df['low'].values
    
    isUp = close_vals > open_vals
    isDown = close_vals < open_vals
    
    obUp = np.full(len(df), False)
    obDown = np.full(len(df), False)
    fvgUp = np.full(len(df), False)
    fvgDown = np.full(len(df), False)
    
    for i in range(2, len(df)):
        if i + 1 < len(df):
            obUp[i] = isDown[i-1] and isUp[i] and close_vals[i] > high_vals[i-1]
            obDown[i] = isUp[i-1] and isDown[i] and close_vals[i] < low_vals[i-1]
        if i + 2 < len(df):
            fvgUp[i] = low_vals[i] > high_vals[i+2]
            fvgDown[i] = high_vals[i] < low_vals[i+2]
    
    obUp = pd.Series(obUp, index=df.index)
    obDown = pd.Series(obDown, index=df.index)
    fvgUp = pd.Series(fvgUp, index=df.index)
    fvgDown = pd.Series(fvgDown, index=df.index)
    
    stacked_bullish = obUp & fvgUp
    stacked_bearish = obDown & fvgDown
    
    # Daily high/low sweep detection
    # Group by day to get daily highs/lows
    day_times = pd.to_datetime(df['time'], unit='s', utc=True).dt.date
    daily_agg = df.groupby(day_times).agg({'high': 'max', 'low': 'min'})
    daily_agg.index = pd.to_datetime(daily_agg.index)
    
    # Get previous day's high/low for each row
    prev_day_high = daily_agg['high'].shift(1)
    prev_day_low = daily_agg['low'].shift(1)
    
    # Map back to original rows
    prev_day_high_map = df['time'].apply(lambda x: prev_day_high.asof(pd.to_datetime(datetime.fromtimestamp(x, tz=timezone.utc).date())))
    prev_day_low_map = df['time'].apply(lambda x: prev_day_low.asof(pd.to_datetime(datetime.fromtimestamp(x, tz=timezone.utc).date())))
    
    # Sweep conditions: price crosses above prev day high or below prev day low
    sweep_high = df['close'] > prev_day_high_map
    sweep_low = df['close'] < prev_day_low_map
    
    # Long entry conditions
    long_condition = crossover_long & time_filter_long & sweep_high
    
    # Short entry conditions  
    short_condition = crossunder_short & time_filter_short & sweep_low
    
    # Iterate and generate entries
    in_position = False
    in_short_position = False
    
    for i in range(2, len(df)):
        if pd.isna(fastEMA.iloc[i]) or pd.isna(slowEMA.iloc[i]):
            continue
            
        if long_condition.iloc[i] and not in_position:
            entry_ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
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
            in_position = True
            
        elif short_condition.iloc[i] and not in_short_position:
            entry_ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
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
            in_short_position = True
        
        # Reset position flags when trend changes
        if crossover_long.iloc[i]:
            in_position = False
        if crossunder_short.iloc[i]:
            in_short_position = False
    
    return entries