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
    # Resample to daily for swing detection
    daily_ohlcv = df.set_index('time').resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    daily_highs = daily_ohlcv['high']
    daily_lows = daily_ohlcv['low']
    
    # Detect swing highs and lows using daily data
    swing_type_list = []
    last_swing_type = None
    for i in range(len(daily_highs)):
        if i >= 3:
            is_swing_high = daily_highs.iloc[i] < daily_highs.iloc[i-1] and daily_highs.iloc[i-3] < daily_highs.iloc[i-2]
            is_swing_low = daily_lows.iloc[i] > daily_lows.iloc[i-1] and daily_lows.iloc[i-3] > daily_lows.iloc[i-2]
            if is_swing_high:
                last_swing_type = "dailyHigh"
            if is_swing_low:
                last_swing_type = "dailyLow"
        swing_type_list.append(last_swing_type)
    
    swing_type_series = pd.Series(swing_type_list, index=daily_highs.index)
    
    # Resample to 4H for FVG detection
    high_4h_resample = df.set_index('time')['high'].resample('4h').max()
    low_4h_resample = df.set_index('time')['low'].resample('4h').min()
    
    high_4h = high_4h_resample.shift(1)
    high_4h_2 = high_4h_resample.shift(3)
    low_4h = low_4h_resample.shift(1)
    low_4h_2 = low_4h_resample.shift(3)
    
    bfvg = low_4h > high_4h_2
    sfvg = high_4h < low_4h_2
    
    result = []
    for i in range(len(df)):
        bar_time = df['time'].iloc[i]
        dt = datetime.fromtimestamp(bar_time, tz=timezone.utc)
        
        if dt.weekday() >= 5:
            continue
        
        window_start_1 = dt.replace(hour=7, minute=45, second=0, microsecond=0)
        window_end_1 = dt.replace(hour=11, minute=45, second=0, microsecond=0)
        window_start_2 = dt.replace(hour=14, minute=0, second=0, microsecond=0)
        window_end_2 = dt.replace(hour=14, minute=45, second=0, microsecond=0)
        
        in_trading_window = (window_start_1 <= dt < window_end_1) or (window_start_2 <= dt < window_end_2)
        
        if not in_trading_window:
            continue
        
        swing_type = swing_type_series.iloc[i] if i < len(swing_type_series) else None
        is_confirmed = True
        
        if bfvg.iloc[i] and is_confirmed and swing_type == "dailyLow":
            result.append({
                'trade_num': len(result) + 1,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
        
        if sfvg.iloc[i] and is_confirmed and swing_type == "dailyHigh":
            result.append({
                'trade_num': len(result) + 1,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
    
    return result