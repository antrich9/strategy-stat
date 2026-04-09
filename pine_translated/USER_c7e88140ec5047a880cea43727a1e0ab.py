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
    
    # Convert time to datetime for daily resampling
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Calculate previous day high and low using daily resampling
    daily_agg = df.set_index('datetime').resample('D').agg({'high': 'max', 'low': 'min'})
    daily_agg['prev_day_high'] = daily_agg['high'].shift(1)
    daily_agg['prev_day_low'] = daily_agg['low'].shift(1)
    daily_agg = daily_agg[['prev_day_high', 'prev_day_low']]
    
    # Merge daily values back to original dataframe
    df = df.merge(daily_agg, left_on='datetime', right_index=True, how='left')
    df['prev_day_high'] = df['prev_day_high'].ffill()
    df['prev_day_low'] = df['prev_day_low'].ffill()
    
    # Detect new day for flag reset
    df['date'] = df['datetime'].dt.date
    df['is_new_day'] = df['date'] != df['date'].shift(1)
    
    # Initialize flags
    flagpdh = False
    flagpdl = False
    
    # Create boolean series for sweep detection
    sweep_pdh = pd.Series(False, index=df.index)
    sweep_pdl = pd.Series(False, index=df.index)
    
    for i in range(1, len(df)):
        if df['is_new_day'].iloc[i]:
            flagpdh = False
            flagpdl = False
        
        if df['close'].iloc[i] > df['prev_day_high'].iloc[i]:
            flagpdh = True
            sweep_pdh.iloc[i] = True
        
        if df['close'].iloc[i] < df['prev_day_low'].iloc[i]:
            flagpdl = True
            sweep_pdl.iloc[i] = True
    
    # Calculate OB conditions
    # obUp: isDown(i+1) and isUp(i) and close[i] > high[i+1]
    is_up_current = df['close'] > df['open']
    is_down_prev = (df['close'].shift(1) < df['open'].shift(1))
    close_above_prev_high = df['close'] > df['high'].shift(1)
    ob_up = is_down_prev & is_up_current & close_above_prev_high
    
    # obDown: isUp(i+1) and isDown(i) and close[i] < low[i+1]
    is_down_current = df['close'] < df['open']
    is_up_prev = (df['close'].shift(1) > df['open'].shift(1))
    close_below_prev_low = df['close'] < df['low'].shift(1)
    ob_down = is_up_prev & is_down_current & close_below_prev_low
    
    # FVG conditions
    # fvgUp: low[i] > high[i+2]
    fvg_up = df['low'] > df['high'].shift(2)
    
    # fvgDown: high[i] < low[i+2]
    fvg_down = df['high'] < df['low'].shift(2)
    
    # Bullish entry: PDH sweep AND stacked OB+FVG
    bullish_entry = sweep_pdh & ob_up & fvg_up
    
    # Bearish entry: PDL sweep AND stacked OB+FVG
    bearish_entry = sweep_pdl & ob_down & fvg_down
    
    # Build entries list
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if bullish_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif bearish_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries