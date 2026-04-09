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
    df['is_new_day'] = pd.to_datetime(df['time'], unit='s').dt.date.diff().ne(0).astype(int)
    df['is_new_day'] = df['is_new_day'].fillna(1).astype(int)
    
    prev_day_high = df['high'].shift(1).where(df['is_new_day'].shift(1) == 1)
    prev_day_low = df['low'].shift(1).where(df['is_new_day'].shift(1) == 1)
    
    for i in range(1, len(df)):
        if df['is_new_day'].iloc[i] == 1:
            prev_day_high.iloc[i] = df['high'].iloc[i-1]
            prev_day_low.iloc[i] = df['low'].iloc[i-1]
        else:
            if pd.isna(prev_day_high.iloc[i]):
                prev_day_high.iloc[i] = prev_day_high.iloc[i-1]
            if pd.isna(prev_day_low.iloc[i]):
                prev_day_low.iloc[i] = prev_day_low.iloc[i-1]
    
    df['prev_day_high'] = prev_day_high
    df['prev_day_low'] = prev_day_low
    
    df['close_above_pdhigh'] = df['close'] > df['prev_day_high']
    df['close_below_pdlow'] = df['close'] < df['prev_day_low']
    
    df['pdh_swept'] = False
    df['pdl_swept'] = False
    
    flagpdh = False
    flagpdl = False
    
    for i in range(1, len(df)):
        if df['is_new_day'].iloc[i] == 1:
            flagpdh = False
            flagpdl = False
        
        if df['close_above_pdhigh'].iloc[i]:
            flagpdh = True
        if df['close_below_pdlow'].iloc[i]:
            flagpdl = True
        
        df.loc[df.index[i], 'pdh_swept'] = flagpdh
        df.loc[df.index[i], 'pdl_swept'] = flagpdl
    
    df['is_up'] = df['close'] > df['open']
    df['is_down'] = df['close'] < df['open']
    
    df['ob_up'] = (
        df['is_down'].shift(1) & 
        df['is_up'] & 
        (df['close'] > df['high'].shift(1))
    )
    df['ob_down'] = (
        df['is_up'].shift(1) & 
        df['is_down'] & 
        (df['close'] < df['low'].shift(1))
    )
    
    df['fvg_up'] = df['low'] > df['high'].shift(2)
    df['fvg_down'] = df['high'] < df['low'].shift(2)
    
    df['stacked_bullish'] = df['ob_up'] & df['fvg_up']
    df['stacked_bearish'] = df['ob_down'] & df['fvg_down']
    
    ema20 = df['close'].ewm(span=20, adjust=False).mean()
    ema50 = df['close'].ewm(span=50, adjust=False).mean()
    ema200 = df['close'].ewm(span=200, adjust=False).mean()
    
    df['ema20'] = ema20
    df['ema50'] = ema50
    df['ema200'] = ema200
    
    low20 = df['low'].rolling(20).min()
    high20 = df['high'].rolling(20).max()
    low50 = df['low'].rolling(50).min()
    high50 = df['high'].rolling(50).max()
    
    df['low20'] = low20
    df['high20'] = high20
    df['low50'] = low50
    df['high50'] = high50
    
    df['bullish_ms'] = (
        (df['close'] > df['high20']) & 
        (df['ema20'] > df['ema50']) &
        (df['ema50'] > df['ema200'])
    )
    df['bearish_ms'] = (
        (df['close'] < df['low20']) & 
        (df['ema20'] < df['ema50']) &
        (df['ema50'] < df['ema200'])
    )
    
    df['waiting_for_long_entry'] = df['pdh_swept'] & df['bullish_ms']
    df['waiting_for_short_entry'] = df['pdl_swept'] & df['bearish_ms']
    
    df['long_entry_cond'] = df['waiting_for_long_entry'] & df['stacked_bullish']
    df['short_entry_cond'] = df['waiting_for_short_entry'] & df['stacked_bearish']
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if i < 3:
            continue
            
        if df['long_entry_cond'].iloc[i]:
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
    
    for i in range(len(df)):
        if i < 3:
            continue
            
        if df['short_entry_cond'].iloc[i]:
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