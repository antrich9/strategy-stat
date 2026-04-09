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
    df = df.copy().reset_index(drop=True)
    
    df['vol_sma'] = df['volume'].rolling(9).mean()
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(np.abs(df['high'] - df['close'].shift(1)), np.abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(20).mean()
    
    df['vol_filt'] = df['volume'] > df['vol_sma'] * 1.5
    df['atr_filt'] = (df['low'] - df['high'].shift(2) > df['atr']) | (df['low'].shift(2) - df['high'] > df['atr'])
    
    df['sma54'] = df['close'].rolling(54).mean()
    df['loc2'] = df['sma54'] > df['sma54'].shift(1)
    
    df['bfvg'] = (df['low'] > df['high'].shift(2)) & df['vol_filt'] & df['atr_filt'] & df['loc2']
    df['sfvg'] = (df['high'] < df['low'].shift(2)) & df['vol_filt'] & df['atr_filt'] & (~df['loc2'])
    
    df['bullish_turn'] = df['bfvg'] & df['sfvg'].shift(1)
    df['bearish_turn'] = df['sfvg'] & df['bfvg'].shift(1)
    
    entries = []
    trade_num = 0
    
    for i in range(2, len(df)):
        if pd.isna(df['close'].iloc[i]) or pd.isna(df['volume'].iloc[i]):
            continue
        
        is_long = df['bullish_turn'].iloc[i] if i < len(df) else False
        is_short = df['bearish_turn'].iloc[i] if i < len(df) else False
        
        if not is_long and not is_short:
            continue
        
        trade_num += 1
        direction = 'long' if is_long else 'short'
        ts = int(df['time'].iloc[i])
        
        entries.append({
            'trade_num': trade_num,
            'direction': direction,
            'entry_ts': ts,
            'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            'entry_price_guess': float(df['close'].iloc[i]),
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': float(df['close'].iloc[i]),
            'raw_price_b': float(df['close'].iloc[i])
        })
    
    return entries