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
    
    if len(df) < 5:
        return entries
    
    # Extract time components for London time window filtering
    # London time: 7:45-9:45 and 14:45-16:45
    dt_series = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert('Europe/London')
    hours = dt_series.dt.hour
    minutes = dt_series.dt.minute
    time_in_minutes = hours * 60 + minutes
    
    # Define London trading windows
    # Morning: 7:45 to 9:45 (465 to 585 minutes)
    # Afternoon: 14:45 to 16:45 (885 to 1005 minutes)
    in_morning_window = (time_in_minutes >= 465) & (time_in_minutes < 585)
    in_afternoon_window = (time_in_minutes >= 885) & (time_in_minutes < 1005)
    in_trading_window = in_morning_window | in_afternoon_window
    
    # Calculate previous day high/low
    prev_day_high = df['high'].shift(1).rolling(window=1440, min_periods=1).max()
    prev_day_low = df['low'].shift(1).rolling(window=1440, min_periods=1).min()
    
    # Volume filter
    volfilt = df['volume'].shift(1) > df['volume'].ewm(span=9, adjust=False).mean() * 1.5
    
    # ATR calculation (Wilder's method)
    high_low = df['high'] - df['low']
    high_close_low = np.abs(df['high'] - df['close'].shift(1))
    low_close_high = np.abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close_low, low_close_high], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/20, adjust=False).mean() / 1.5
    atrfilt = (df['low'] - df['high'].shift(2) > atr) | (df['low'].shift(2) - df['high'] > atr)
    
    # Trend filter
    loc = df['close'].ewm(span=54, adjust=False).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    # OB/FVG conditions
    close_arr = df['close'].values
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    
    def is_up(idx):
        return close_arr[idx] > open_arr[idx]
    
    def is_down(idx):
        return close_arr[idx] < open_arr[idx]
    
    def is_ob_up(idx):
        return is_down(idx + 1) and is_up(idx) and close_arr[idx] > high_arr[idx + 1]
    
    def is_ob_down(idx):
        return is_up(idx + 1) and is_down(idx) and close_arr[idx] < low_arr[idx + 1]
    
    def is_fvg_up(idx):
        return low_arr[idx] > high_arr[idx + 2]
    
    def is_fvg_down(idx):
        return high_arr[idx] < low_arr[idx + 2]
    
    ob_up = pd.Series([is_ob_up(i) if i >= 2 else False for i in range(len(df))], index=df.index)
    ob_down = pd.Series([is_ob_down(i) if i >= 2 else False for i in range(len(df))], index=df.index)
    fvg_up = pd.Series([is_fvg_up(i) if i >= 2 else False for i in range(len(df))], index=df.index)
    fvg_down = pd.Series([is_fvg_down(i) if i >= 2 else False for i in range(len(df))], index=df.index)
    
    # Breakaway FVG
    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts
    
    # Crossover/crossunder for indicators
    ema_fast = df['close'].ewm(span=9, adjust=False).mean()
    ema_slow = df['close'].ewm(span=21, adjust=False).mean()
    
    # Entry conditions
    long_condition = in_trading_window & (ob_up | fvg_up | bfvg)
    short_condition = in_trading_window & (ob_down | fvg_down | sfvg)
    
    for i in range(5, len(df)):
        if np.isnan(df['close'].iloc[i]) or np.isnan(atr.iloc[i]):
            continue
        
        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        entry_price = float(df['close'].iloc[i])
        
        if long_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
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