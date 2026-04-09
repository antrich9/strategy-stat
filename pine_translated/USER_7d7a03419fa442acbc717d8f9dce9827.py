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
    
    # Convert to UTC timestamps and extract datetime components for filtering
    df = df.copy()
    df['ts'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = df['ts'].dt.hour
    df['date'] = df['ts'].dt.date
    
    # Define trading windows in London time (UTC during winter, BST+1 during summer)
    # Window 1: 07:00-11:45, Window 2: 14:00-14:45
    def in_trading_window(row):
        hour = row['hour']
        # Account for London timezone offset (0 or 1)
        return (7 <= hour < 12) or (14 <= hour < 15)
    
    df['in_window'] = df.apply(in_trading_window, axis=1)
    
    # Extract 4-hour OHLC data by resampling
    resampled_4h = df.set_index('ts').resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    # Extract daily OHLC data by resampling
    resampled_daily = df.set_index('ts').resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    # Calculate 4-hour high and low for the previous two periods
    high_4h = resampled_4h['high']
    low_4h = resampled_4h['low']
    
    # For FVG detection, get the high and low from two 4-hour periods back
    high_4h_2 = high_4h.shift(2)
    low_4h_2 = low_4h.shift(2)
    
    # Calculate daily high and low
    daily_high = resampled_daily['high']
    daily_low = resampled_daily['low']
    
    # Get the previous day's high and low
    prev_daily_high = daily_high.shift(1)
    prev_daily_low = daily_low.shift(1)
    
    # Get daily high and low from two days back
    daily_high_2 = daily_high.shift(2)
    daily_low_2 = daily_low.shift(2)
    
    # Identify bullish swing low: prior day's low exceeds the low from two days prior
    is_bullish_swing = (prev_daily_low > daily_low_2)
    
    # Identify bearish swing high: prior day's high is lower than the high from two days prior
    is_bearish_swing = (prev_daily_high < daily_high_2)
    
    # Detect bullish FVG: current 4-hour low surpasses the high from two periods back
    bfvg = low_4h > high_4h_2
    
    # Detect bearish FVG: current 4-hour high falls below the low from two periods back
    sfvg = high_4h < low_4h_2
    
    # Build a combined DataFrame with all computed values, then merge with original data using forward-fill for alignment
    indicators = pd.DataFrame({
        'ts': resampled_4h.index,
        'high_4h': high_4h.values,
        'low_4h': low_4h.values,
        'high_4h_2': high_4h_2.values,
        'low_4h_2': low_4h_2.values,
        'bfvg': bfvg.values,
        'sfvg': sfvg.values,
        'is_bullish_swing': is_bullish_swing.values,
        'is_bearish_swing': is_bearish_swing.values
    })
    
    df = df.merge(indicators, on='ts', how='left')
    df.ffill(inplace=True)
    df.dropna(subset=['bfvg', 'sfvg', 'is_bullish_swing', 'is_bearish_swing'], inplace=True)
    
    # Bullish entry triggers when BFVG aligns with bullish swing and price is within the trading window
    df['bull_entry'] = df['bfvg'] & df['is_bullish_swing'] & df['in_window']
    
    # Bearish entry triggers when SFVG aligns with bearish swing and price is within the trading window
    df['bear_entry'] = df['sfvg'] & df['is_bearish_swing'] & df['in_window']
    
    trades = []
    trade_num = 1
    
    for idx, row in df.iterrows():
        if row['bull_entry']:
            trades.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(row['time']),
                'entry_time': datetime.fromtimestamp(row['time'], tz=timezone.utc).isoformat(),
                'entry_price_guess': row['close'],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': row['close'],
                'raw_price_b': row['close']
            })
            trade_num += 1
        elif row['bear_entry']:
            trades.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(row['time']),
                'entry_time': datetime.fromtimestamp(row['time'], tz=timezone.utc).isoformat(),
                'entry_price_guess': row['close'],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': row['close'],
                'raw_price_b': row['close']
            })
            trade_num += 1
    
    return trades