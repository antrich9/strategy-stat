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
    df['ts'] = df['time']
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    
    london_morning_start = (df['hour'] == 7) & (df['minute'] >= 45) | (df['hour'] > 7) & (df['hour'] < 9) | (df['hour'] == 9) & (df['minute'] < 45)
    london_afternoon_start = (df['hour'] == 14) & (df['minute'] >= 45) | (df['hour'] > 14) & (df['hour'] < 16) | (df['hour'] == 16) & (df['minute'] < 45)
    df['in_trading_window'] = london_morning_start | london_afternoon_start
    
    df['is_up'] = df['close'] > df['open']
    df['is_down'] = df['close'] < df['open']
    
    df['ob_up'] = df['is_down'].shift(1) & df['is_up'] & (df['close'] > df['high'].shift(1))
    df['ob_down'] = df['is_up'].shift(1) & df['is_down'] & (df['close'] < df['low'].shift(1))
    
    df['fvg_up'] = df['low'] > df['high'].shift(2)
    df['fvg_down'] = df['high'] < df['low'].shift(2)
    
    df['bull_stack'] = df['fvg_up'] & df['ob_up'].shift(1)
    df['bear_stack'] = df['fvg_down'] & df['ob_down'].shift(1)
    
    df['long_entry'] = df['bull_stack'] & df['in_trading_window']
    df['short_entry'] = df['bear_stack'] & df['in_trading_window']
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(df['fvg_up'].iloc[i]) or pd.isna(df['ob_up'].iloc[i]) or pd.isna(df['in_trading_window'].iloc[i]):
            continue
        
        direction = None
        if df['long_entry'].iloc[i]:
            direction = 'long'
        elif df['short_entry'].iloc[i]:
            direction = 'short'
        
        if direction:
            ts = int(df['ts'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries