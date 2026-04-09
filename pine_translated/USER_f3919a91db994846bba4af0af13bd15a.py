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
    
    # Ensure we have necessary columns
    required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            return []
    
    df = df.copy()
    df['hour'] = df['time'].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc).hour)
    
    # Helper functions for OB and FVG detection
    def is_up(idx):
        return df['close'].iloc[idx] > df['open'].iloc[idx]
    
    def is_down(idx):
        return df['close'].iloc[idx] < df['open'].iloc[idx]
    
    def is_ob_up(idx):
        return (is_down(idx + 1) and is_up(idx) and 
                df['close'].iloc[idx] > df['high'].iloc[idx + 1])
    
    def is_ob_down(idx):
        return (is_up(idx + 1) and is_down(idx) and 
                df['close'].iloc[idx] < df['low'].iloc[idx + 1])
    
    def is_fvg_up(idx):
        return (df['low'].iloc[idx] > df['high'].iloc[idx + 2])
    
    def is_fvg_down(idx):
        return (df['high'].iloc[idx] < df['low'].iloc[idx + 2])
    
    # Time filter
    df['isValidTradeTime'] = ((df['hour'] >= 2) & (df['hour'] < 5)) | ((df['hour'] >= 10) & (df['hour'] < 12))
    
    # Volume filter
    df['volfilt'] = df['volume'].shift(1) > df['volume'].rolling(9).mean() * 1.5
    
    # ATR filter (Wilder ATR)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean() / 1.5
    df['atrfilt'] = ((df['low'] - df['high'].shift(2) > atr) | (df['low'].shift(2) - df['high'] > atr))
    
    # Trend filter
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    df['locfiltb'] = loc2
    df['locfilts'] = ~loc2
    
    # FVG conditions
    df['bfvg'] = (df['low'] > df['high'].shift(2)) & df['volfilt'] & df['atrfilt'] & df['locfiltb']
    df['sfvg'] = (df['high'] < df['low'].shift(2)) & df['volfilt'] & df['atrfilt'] & df['locfilts']
    
    # Stacked OB + FVG conditions (checking at bar i for conditions that formed at i-1 and i)
    df['obUp'] = df.index.to_series().apply(lambda i: is_ob_up(i - 1) if i >= 2 else False)
    df['obDown'] = df.index.to_series().apply(lambda i: is_ob_down(i - 1) if i >= 2 else False)
    df['fvgUp'] = df.index.to_series().apply(lambda i: is_fvg_up(i) if i >= 2 else False)
    df['fvgDown'] = df.index.to_series().apply(lambda i: is_fvg_down(i) if i >= 2 else False)
    
    # Entry conditions: stacked OB + FVG with valid time
    df['bullish_entry'] = df['obUp'] & df['fvgUp'] & df['isValidTradeTime']
    df['bearish_entry'] = df['obDown'] & df['fvgDown'] & df['isValidTradeTime']
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(df['bullish_entry'].iloc[i]) or pd.isna(df['bearish_entry'].iloc[i]):
            continue
        if df['bullish_entry'].iloc[i]:
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
            trade_num += 1
        elif df['bearish_entry'].iloc[i]:
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
            trade_num += 1
    
    return entries