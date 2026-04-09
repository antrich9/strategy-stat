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

    if len(df) < 3:
        return results

    # === Previous Day High/Low Calculation ===
    df['date'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.date
    df['prev_date'] = df['date'].shift(1)
    
    # Group by date to get daily high/low
    daily_agg = df.groupby('date').agg({'high': 'max', 'low': 'min'}).reset_index()
    daily_agg.columns = ['date', 'daily_high', 'daily_low']
    daily_agg['prev_daily_high'] = daily_agg['daily_high'].shift(1)
    daily_agg['prev_daily_low'] = daily_agg['daily_low'].shift(1)
    
    df = df.merge(daily_agg[['date', 'prev_daily_high', 'prev_daily_low']], on='date', how='left')
    
    # === New Day Detection ===
    df['new_day'] = df['date'] != df['date'].shift(1)
    
    # === Bias State Tracking ===
    bias = 0
    last_fvg = 0  # 1 = Bullish, -1 = Bearish, 0 = None
    
    # Forward fill previous day high/low
    df['pdh'] = df['prev_daily_high'].ffill()
    df['pdl'] = df['prev_daily_low'].ffill()
    
    # Fill NaN values with first valid values
    df['pdh'] = df['pdh'].fillna(method='ffill')
    df['pdl'] = df['pdl'].fillna(method='ffill')
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        
        # Skip if required values are NaN
        if pd.isna(row['pdh']) or pd.isna(row['pdl']):
            continue
        
        pdh = row['pdh']
        pdl = row['pdl']
        
        # Reset bias on new day
        if row['new_day']:
            bias = 0
        
        # Current bar conditions
        high_val = row['high']
        low_val = row['low']
        close_val = row['close']
        
        swept_low = low_val < pdl
        swept_high = high_val > pdh
        broke_high = close_val > pdh
        broke_low = close_val < pdl
        
        # Priority logic for bias
        if swept_low and broke_high:
            bias = 1
        elif swept_high and broke_low:
            bias = -1
        elif low_val < pdl:
            bias = -1
        elif high_val > pdh:
            bias = 1
        
        # FVG Detection using 4H data (simulated from available data)
        # For 4H candles, we need to look at bars at 4H intervals
        # Using approximations based on available data
        # This is a simplified version - in real implementation, 4H data would be fetched
        
        # For this conversion, we'll detect FVGs based on available bar data
        # A bullish FVG forms when: low > high[2] (gap up from 2 bars ago)
        # A bearish FVG forms when: high < low[2] (gap down from 2 bars ago)
        
        if i >= 2:
            prev_high_1 = df.iloc[i-1]['high']
            prev_low_1 = df.iloc[i-1]['low']
            prev_high_2 = df.iloc[i-2]['high']
            prev_low_2 = df.iloc[i-2]['low']
            
            current_low = low_val
            current_high = high_val
            
            bullish_fvg = current_low > prev_high_2
            bearish_fvg = current_high < prev_low_2
            
            # Sharp turn detection - entry signal
            if bullish_fvg and last_fvg == -1:
                # Bullish Sharp Turn - Long Entry
                ts = int(row['time'])
                entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                
                results.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': entry_time,
                    'entry_price_guess': close_val,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close_val,
                    'raw_price_b': close_val
                })
                trade_num += 1
                last_fvg = 1
                
            elif bearish_fvg and last_fvg == 1:
                # Bearish Sharp Turn - Short Entry
                ts = int(row['time'])
                entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                
                results.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': ts,
                    'entry_time': entry_time,
                    'entry_price_guess': close_val,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close_val,
                    'raw_price_b': close_val
                })
                trade_num += 1
                last_fvg = -1
                
            elif bullish_fvg:
                last_fvg = 1
            elif bearish_fvg:
                last_fvg = -1
    
    return results