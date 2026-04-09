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
    # Convert time to datetime for grouping by day
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['day'] = df['datetime'].dt.date
    
    # Calculate rolling previous day high and low
    # Group by day and take the high/low, then shift to get previous day values
    daily_hl = df.groupby('day')['high'].max().reset_index()
    daily_hl.columns = ['day', 'day_high']
    daily_ll = df.groupby('day')['low'].min().reset_index()
    daily_ll.columns = ['day', 'day_low']
    
    # Merge daily values back
    df = df.merge(daily_hl, on='day', how='left')
    df = df.merge(daily_ll, on='day', how='left')
    
    # Shift to get previous day high/low for each row
    # For each bar, prev_day_high is the high of the PREVIOUS calendar day
    prev_day_high_arr = df.groupby('day')['day_high'].first().shift(1).reindex(df['day']).values
    prev_day_low_arr = df.groupby('day')['day_low'].first().shift(1).reindex(df['day']).values
    
    # Define conditions as boolean Series
    close = df['close']
    open_arr = df['open']
    high = df['high']
    low = df['low']
    
    # obUp: close[1] > open[1] AND close[2] < open[2] AND close[1] > high[2]
    # In Python with i as current bar: close[i] > open[i] AND close[i+1] < open[i+1] AND close[i] > high[i+1]
    ob_up = (close > open_arr) & (close.shift(1) < open_arr.shift(1)) & (close > high.shift(1))
    
    # fvgUp: low[0] > high[2]
    # In Python with i as current bar: low[i] > high[i+2]
    fvg_up = low > high.shift(2)
    
    # Bullish entry condition: close > prevDayHigh AND obUp AND fvgUp
    bullish_entry = (close > pd.Series(prev_day_high_arr, index=df.index)) & ob_up & fvg_up
    
    # Iterate and generate entries
    trades = []
    trade_num = 0
    
    # Need at least 3 bars for fvgUp check (i, i+1, i+2)
    max_idx = len(df) - 3
    
    for i in range(max_idx + 1):
        # Skip if required indicators are NaN
        if pd.isna(prev_day_high_arr[i]) or pd.isna(prev_day_low_arr[i]):
            continue
        
        if bullish_entry.iloc[i]:
            trade_num += 1
            ts = df['time'].iloc[i]
            entry_price = df['close'].iloc[i]
            trades.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
    
    return trades