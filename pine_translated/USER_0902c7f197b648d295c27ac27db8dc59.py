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
    df = df.copy()
    df['dt'] = pd.to_datetime(df['time'], unit='s')
    df['hour'] = df['dt'].dt.hour
    df['minute'] = df['dt'].dt.minute
    df['time_minutes'] = df['hour'] * 60 + df['minute']
    
    morning_window = (df['time_minutes'] >= 465) & (df['time_minutes'] < 585)
    afternoon_window = (df['time_minutes'] >= 885) & (df['time_minutes'] < 1005)
    in_trading_window = morning_window | afternoon_window
    
    prev_day_high = df['high'].rolling(96).max().shift(1)
    prev_day_low = df['low'].rolling(96).min().shift(1)
    
    swing_high = (df['high'].shift(3) < df['high'].shift(2)) & \
                 (df['high'].shift(1) <= df['high'].shift(2)) & \
                 (df['high'].shift(2) >= df['high'].shift(4)) & \
                 (df['high'].shift(2) >= df['high'].shift(5))
    
    swing_low = (df['low'].shift(3) > df['low'].shift(2)) & \
                (df['low'].shift(1) >= df['low'].shift(2)) & \
                (df['low'].shift(2) <= df['low'].shift(4)) & \
                (df['low'].shift(2) <= df['low'].shift(5))
    
    df['last_swing_high'] = np.where(swing_high, df['high'].shift(2), np.nan)
    df['last_swing_high'] = df['last_swing_high'].ffill()
    df['last_swing_low'] = np.where(swing_low, df['low'].shift(2), np.nan)
    df['last_swing_low'] = df['last_swing_low'].ffill()
    
    bullish_count = swing_high.cumsum()
    bearish_count = swing_low.cumsum()
    
    prev_day_high_taken = df['high'] > prev_day_high
    prev_day_low_taken = df['low'] < prev_day_low
    
    flag_pdh = prev_day_high_taken
    flag_pdl = prev_day_low_taken
    
    entries = []
    trade_num = 0
    
    for i in range(5, len(df)):
        if in_trading_window.iloc[i] and flag_pdh.iloc[i]:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
        elif in_trading_window.iloc[i] and flag_pdl.iloc[i]:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
    
    return entries