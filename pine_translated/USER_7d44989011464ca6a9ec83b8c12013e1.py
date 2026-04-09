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
    
    # Helper functions
    def wilder_rsi(series, length):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def wilder_atr(high, low, close, length):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/length, adjust=False).mean()
        return atr
    
    def crossover(a, b):
        return (a > b) & (a.shift(1) <= b.shift(1))
    
    def crossunder(a, b):
        return (a < b) & (a.shift(1) >= b.shift(1))
    
    # Convert time to datetime and resample to 4H
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.set_index('datetime')
    
    # Resample to 4H candles
    h4_data = df.resample('240min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'time': 'first'
    }).dropna(subset=['high', 'low', 'close'])
    
    if len(h4_data) < 3:
        return []
    
    h4_data = h4_data.reset_index(drop=True)
    h4_data.index = range(len(h4_data))
    
    # Forward fill to handle any gaps
    for col in ['high', 'low', 'close', 'volume', 'time']:
        h4_data[col] = h4_data[col].ffill().bfill()
    
    high_4h = h4_data['high']
    low_4h = h4_data['low']
    close_4h = h4_data['close']
    volume_4h = h4_data['volume']
    
    # Volume Filter (4H)
    vol_sma = volume_4h.rolling(9).mean()
    volfilt1 = volume_4h.shift(1) > vol_sma * 1.5
    
    # ATR Filter (4H)
    atr_4h = wilder_atr(high_4h, low_4h, close_4h, 20) / 1.5
    atrfilt1 = (low_4h - high_4h.shift(2) > atr_4h) | (low_4h.shift(2) - high_4h > atr_4h)
    
    # Trend Filter (4H)
    loc1 = close_4h.rolling(54).mean()
    loc21 = loc1 > loc1.shift(1)
    locfiltb1 = loc21
    locfilts1 = ~loc21
    
    # Bullish and Bearish FVGs (4H)
    bfvg1 = (low_4h > high_4h.shift(2)) & volfilt1 & atrfilt1 & locfiltb1
    sfvg1 = (high_4h < low_4h.shift(2)) & volfilt1 & atrfilt1 & locfilts1
    
    # Identify new 4H bars
    is_new_4h = pd.Series(True, index=h4_data.index)
    is_new_4h.iloc[0] = True
    
    # Track last FVG type
    last_fvg = 0
    trade_num = 1
    entries = []
    
    # Entry conditions
    long_condition = bfvg1 & (last_fvg == -1)
    short_condition = sfvg1 & (last_fvg == 1)
    
    for i in range(len(h4_data)):
        if i == 0 or is_new_4h.iloc[i]:
            if long_condition.iloc[i] and not pd.isna(bfvg1.iloc[i]):
                last_fvg = 1
            elif short_condition.iloc[i] and not pd.isna(sfvg1.iloc[i]):
                last_fvg = -1
            elif bfvg1.iloc[i] and not pd.isna(bfvg1.iloc[i]):
                last_fvg = 1
            elif sfvg1.iloc[i] and not pd.isna(sfvg1.iloc[i]):
                last_fvg = -1
    
    # Clear and rebuild entries
    last_fvg = 0
    trade_num = 1
    entries = []
    
    for i in range(len(h4_data)):
        if i == 0 or is_new_4h.iloc[i]:
            # Check for bullish sharp turn (long entry)
            if long_condition.iloc[i] and not pd.isna(bfvg1.iloc[i]):
                entry_price = close_4h.iloc[i]
                entry_ts = int(h4_data['time'].iloc[i])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
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
                last_fvg = 1
            # Check for bearish sharp turn (short entry)
            elif short_condition.iloc[i] and not pd.isna(sfvg1.iloc[i]):
                entry_price = close_4h.iloc[i]
                entry_ts = int(h4_data['time'].iloc[i])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
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
                last_fvg = -1
            # Update last_fvg for regular FVGs
            elif bfvg1.iloc[i] and not pd.isna(bfvg1.iloc[i]):
                last_fvg = 1
            elif sfvg1.iloc[i] and not pd.isna(sfvg1.iloc[i]):
                last_fvg = -1
    
    return entries