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
    
    open_prices = df['open']
    high_prices = df['high']
    low_prices = df['low']
    close_prices = df['close']
    
    # Helper functions for OB and FVG conditions
    def is_up(idx):
        return close_prices.iloc[idx] > open_prices.iloc[idx]
    
    def is_down(idx):
        return close_prices.iloc[idx] < open_prices.iloc[idx]
    
    def is_ob_up(idx):
        # Bullish OB: down candle followed by up candle, close > previous high
        return (is_down(idx - 1) and is_up(idx) and 
                close_prices.iloc[idx] > high_prices.iloc[idx - 1])
    
    def is_ob_down(idx):
        # Bearish OB: up candle followed by down candle, close < previous low
        return (is_up(idx - 1) and is_down(idx) and 
                close_prices.iloc[idx] < low_prices.iloc[idx - 1])
    
    def is_fvg_up(idx):
        # Bullish FVG: current low > high from 2 bars ago
        return low_prices.iloc[idx] > high_prices.iloc[idx - 2]
    
    def is_fvg_down(idx):
        # Bearish FVG: current high < low from 2 bars ago
        return high_prices.iloc[idx] < low_prices.iloc[idx - 2]
    
    # Calculate previous day high and low using rolling daily high/low
    # Approximation: use 24-period rolling max/min as proxy for daily levels
    prev_day_high = high_prices.rolling(window=24, min_periods=24).max().shift(1)
    prev_day_low = low_prices.rolling(window=24, min_periods=24).min().shift(1)
    
    # Detect sweeps
    pdh_swept = close_prices > prev_day_high
    pdl_swept = close_prices < prev_day_low
    
    # Initialize flags
    flagpdh = pd.Series(False, index=df.index)
    flagpdl = pd.Series(False, index=df.index)
    
    # Track sweep flags
    for i in range(len(df)):
        if i > 0:
            if pdh_swept.iloc[i]:
                flagpdh.iloc[i] = True
            else:
                flagpdh.iloc[i] = flagpdh.iloc[i-1]
                
            if pdl_swept.iloc[i]:
                flagpdl.iloc[i] = True
            else:
                flagpdl.iloc[i] = flagpdl.iloc[i-1]
        else:
            if pdh_swept.iloc[i]:
                flagpdh.iloc[i] = True
            if pdl_swept.iloc[i]:
                flagpdl.iloc[i] = True
    
    # Calculate OB and FVG conditions for each bar
    ob_up = pd.Series(False, index=df.index)
    ob_down = pd.Series(False, index=df.index)
    fvg_up = pd.Series(False, index=df.index)
    fvg_down = pd.Series(False, index=df.index)
    
    for i in range(2, len(df)):
        try:
            ob_up.iloc[i] = is_ob_up(i)
            ob_down.iloc[i] = is_ob_down(i)
            fvg_up.iloc[i] = is_fvg_up(i)
            fvg_down.iloc[i] = is_fvg_down(i)
        except:
            pass
    
    # Entry conditions
    long_condition = flagpdh & ob_up & fvg_up
    short_condition = flagpdl & ob_down & fvg_down
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if i < 2:
            continue
            
        if long_condition.iloc[i] and not long_condition.iloc[i-1]:
            entry_price = close_prices.iloc[i]
            entry_ts = int(df['time'].iloc[i])
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
            
        if short_condition.iloc[i] and not short_condition.iloc[i-1]:
            entry_price = close_prices.iloc[i]
            entry_ts = int(df['time'].iloc[i])
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
    
    return entries