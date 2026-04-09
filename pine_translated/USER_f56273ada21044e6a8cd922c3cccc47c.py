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
    
    # Ensure required columns exist
    required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            return []
    
    # Filter for London trading windows: 07:45-09:45 and 15:45-16:45 UTC
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    
    # Morning window: 07:45 to 09:45
    morning_start = (df['hour'] == 7) & (df['minute'] >= 45)
    morning_mid = (df['hour'] == 8)
    morning_end = (df['hour'] == 9) & (df['minute'] <= 44)
    isWithinMorningWindow = morning_start | morning_mid | morning_end
    
    # Afternoon window: 15:45 to 16:45
    afternoon_start = (df['hour'] == 15) & (df['minute'] >= 45)
    afternoon_mid = (df['hour'] == 16) & (df['minute'] <= 44)
    isWithinAfternoonWindow = afternoon_start | afternoon_mid
    
    in_trading_window = isWithinMorningWindow | isWithinAfternoonWindow
    
    # Get previous day high/low - use daily resample
    df_with_idx = df.set_index('datetime')
    daily_agg = df_with_idx.resample('D').agg({'high': 'max', 'low': 'min'}).dropna()
    
    # Shift to get previous day values
    daily_agg['prev_day_high'] = daily_agg['high'].shift(1)
    daily_agg['prev_day_low'] = daily_agg['low'].shift(1)
    
    # Forward fill prev day values to align with intraday data
    daily_agg['prev_day_high'] = daily_agg['prev_day_high'].ffill()
    daily_agg['prev_day_low'] = daily_agg['prev_day_low'].ffill()
    
    # Merge back to intraday df
    df['date'] = df['datetime'].dt.date
    daily_reset = daily_agg[['prev_day_high', 'prev_day_low']].reset_index()
    daily_reset['date'] = daily_reset['index'].dt.date
    df = df.merge(daily_reset[['date', 'prev_day_high', 'prev_day_low']], on='date', how='left')
    
    # Forward fill within day
    df['prev_day_high'] = df['prev_day_high'].ffill()
    df['prev_day_low'] = df['prev_day_low'].ffill()
    
    # Get current day high/low for 240 timeframe equivalent (use rolling max/min across 4 bars approximation)
    # For simplicity, use rolling 6-hour window (assuming 1-hour bars for 240 TF equivalent)
    df['current_day_high'] = df['high'].rolling(window=6, min_periods=1).max()
    df['current_day_low'] = df['low'].rolling(window=6, min_periods=1).min()
    
    # Detect previous day high/low sweeps
    df['previousDayHighTaken'] = df['high'] > df['prev_day_high']
    df['previousDayLowTaken'] = df['low'] < df['prev_day_low']
    
    # Track flags - previous bar values
    df['flagpdh'] = False
    df['flagpdl'] = False
    
    # Build entry conditions based on the visible logic:
    # flagpdh = previousDayHighTaken and currentDayLow > prevDayLow
    # flagpdl = previousDayLowTaken and currentDayHigh < prevDayHigh
    
    for i in range(1, len(df)):
        cond1 = df['previousDayHighTaken'].iloc[i] and df['current_day_low'].iloc[i] > df['prev_day_low'].iloc[i]
        cond2 = df['previousDayLowTaken'].iloc[i] and df['current_day_high'].iloc[i] < df['prev_day_high'].iloc[i]
        
        if cond1:
            df.loc[df.index[i], 'flagpdh'] = True
        elif cond2:
            df.loc[df.index[i], 'flagpdl'] = True
    
    # Entry conditions: within trading window AND (flagpdh for long OR flagpdl for short)
    df['long_condition'] = in_trading_window & df['flagpdh']
    df['short_condition'] = in_trading_window & df['flagpdl']
    
    entries = []
    trade_num = 1
    
    for i in range(1, len(df)):
        if df['long_condition'].iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif df['short_condition'].iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries