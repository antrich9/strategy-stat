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
    
    # Convert time to datetime for filtering
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    
    # Time window: 09:45 - 10:45 (London time, which is UTC in winter, UTC+1 in summer)
    # For simplicity, we check if hour is 9 or 10 with specific minute conditions
    in_window = ((df['hour'] == 9) & (df['minute'] >= 45)) | ((df['hour'] == 10) & (df['minute'] < 45))
    
    # Resample to 4H timeframe
    df['time_4h'] = df['time'].floordiv(14400000) * 14400000  # Floor to 4H boundaries
    df_4h = df.groupby('time_4h').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'open': 'first',
        'volume': 'sum',
        'datetime': 'last'
    }).reset_index()
    
    # Detect new 4H candle
    df_4h['is_new_4h'] = df_4h['time_4h'] != df_4h['time_4h'].shift(1)
    
    # Volume Filter - using 4H data
    vol_sma = df_4h['volume'].rolling(9).mean()
    df_4h['volfilt'] = df_4h['volume'].shift(1) > vol_sma.shift(1) * 1.5
    
    # ATR Filter - using 4H data (Wilder ATR)
    high = df_4h['high']
    low = df_4h['low']
    close_prev = df_4h['close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_length = 20
    atr = tr.ewm(alpha=1/atr_length, adjust=False).mean()
    df_4h['atr_4h'] = atr / 1.5
    
    df_4h['atrfilt'] = ((low - high.shift(2) > df_4h['atr_4h']) | (low.shift(2) - high > df_4h['atr_4h']))
    
    # Trend Filter - using 4H data
    loc1 = df_4h['close'].rolling(54).mean()
    loc21 = loc1 > loc1.shift(1)
    df_4h['locfiltb'] = loc21
    df_4h['locfilts'] = ~loc21
    
    # Identify Bullish and Bearish FVGs - using 4H data
    df_4h['bfvg'] = (low > high.shift(2)) & df_4h['volfilt'] & df_4h['atrfilt'] & df_4h['locfiltb']
    df_4h['sfvg'] = (high < low.shift(2)) & df_4h['volfilt'] & df_4h['atrfilt'] & df_4h['locfilts']
    
    # Track last FVG type
    lastFVG = 0
    latest_4h_time = 0
    
    # Iterate through bars to detect entries
    for i in range(len(df)):
        row = df.iloc[i]
        current_4h_time = row['time_4h']
        
        # Find corresponding 4H data
        idx_4h = df_4h[df_4h['time_4h'] == current_4h_time].index
        if len(idx_4h) == 0:
            continue
        idx_4h = idx_4h[0]
        
        # Check if it's a new 4H candle
        is_new_4h = df_4h.iloc[idx_4h]['is_new_4h'] if idx_4h > 0 else True
        
        if is_new_4h:
            if df_4h.iloc[idx_4h]['bfvg'] and lastFVG == -1:
                # Bullish Sharp Turn - Long Entry
                if in_window.iloc[i]:
                    entry = {
                        'trade_num': trade_num,
                        'direction': 'long',
                        'entry_ts': int(row['time']),
                        'entry_time': datetime.fromtimestamp(row['time'], tz=timezone.utc).isoformat(),
                        'entry_price_guess': float(row['close']),
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': float(row['close']),
                        'raw_price_b': float(row['close'])
                    }
                    entries.append(entry)
                    trade_num += 1
                lastFVG = 1
            elif df_4h.iloc[idx_4h]['sfvg'] and lastFVG == 1:
                # Bearish Sharp Turn - Short Entry
                if in_window.iloc[i]:
                    entry = {
                        'trade_num': trade_num,
                        'direction': 'short',
                        'entry_ts': int(row['time']),
                        'entry_time': datetime.fromtimestamp(row['time'], tz=timezone.utc).isoformat(),
                        'entry_price_guess': float(row['close']),
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': float(row['close']),
                        'raw_price_b': float(row['close'])
                    }
                    entries.append(entry)
                    trade_num += 1
                lastFVG = -1
            elif df_4h.iloc[idx_4h]['bfvg']:
                lastFVG = 1
            elif df_4h.iloc[idx_4h]['sfvg']:
                lastFVG = -1
    
    return entries