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
    df['time_dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = df['time_dt'].dt.hour
    df['minute'] = df['time_dt'].dt.minute
    df['time_minutes'] = df['hour'] * 60 + df['minute']
    df['month'] = df['time_dt'].dt.month
    df['day'] = df['time_dt'].dt.day
    df['year'] = df['time_dt'].dt.year
    
    london_offset = df['time_dt'].dt.month.apply(lambda m: 60 if m >= 3 and m <= 10 else 0)
    df['london_time_minutes'] = (df['time_minutes'] + london_offset) % 1440
    
    london_start_window1 = 7 * 60
    london_end_window1 = 11 * 60 + 45
    london_start_window2 = 14 * 60
    london_end_window2 = 14 * 60 + 45
    
    df['isWithinWindow1'] = (df['london_time_minutes'] >= london_start_window1) & (df['london_time_minutes'] < london_end_window1)
    df['isWithinWindow2'] = (df['london_time_minutes'] >= london_start_window2) & (df['london_time_minutes'] < london_end_window2)
    df['in_trading_window'] = df['isWithinWindow1'] | df['isWithinWindow2']
    
    volfilt = (df['volume'].shift(1) > df['volume'].rolling(9).mean() * 1.5)
    df['volfilt'] = volfilt.fillna(True)
    
    atr_length = 20
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1.0/atr_length, adjust=False).mean()
    df['atr2'] = atr / 1.5
    
    df['atrfilt'] = ((df['low'] - df['high'].shift(2) > df['atr2']) | (df['low'].shift(2) - df['high'] > df['atr2']))
    
    df['loc'] = df['close'].rolling(54).mean()
    df['loc2'] = df['loc'] > df['loc'].shift(1)
    df['locfiltb'] = df['loc2']
    df['locfilts'] = ~df['loc2']
    
    df['bfvg'] = (df['low'] > df['high'].shift(2)) & df['volfilt'] & df['atrfilt'] & df['locfiltb']
    df['sfvg'] = (df['high'] < df['low'].shift(2)) & df['volfilt'] & df['atrfilt'] & df['locfilts']
    
    df['lastFVG'] = 0
    for i in range(1, len(df)):
        if df['bfvg'].iloc[i]:
            df.iloc[i, df.columns.get_loc('lastFVG')] = 1
        elif df['sfvg'].iloc[i]:
            df.iloc[i, df.columns.get_loc('lastFVG')] = -1
        else:
            df.iloc[i, df.columns.get_loc('lastFVG')] = df['lastFVG'].iloc[i-1]
    
    df['prev_lastFVG'] = df['lastFVG'].shift(1)
    
    df['bullish_sharp_turn'] = df['bfvg'] & (df['prev_lastFVG'] == -1)
    df['bearish_sharp_turn'] = df['sfvg'] & (df['prev_lastFVG'] == 1)
    
    entries = []
    trade_num = 1
    
    for i in range(1, len(df)):
        if df['in_trading_window'].iloc[i] and df['bullish_sharp_turn'].iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            
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
            
        elif df['in_trading_window'].iloc[i] and df['bearish_sharp_turn'].iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            
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