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
    
    # Settings from Pine Script (using defaults)
    filterWidth = 0.0
    showBull = True
    showBear = True
    
    # Calculate Wilder ATR(200)
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range calculation
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Wilder ATR(200) - Wilder smoothing
    atr = pd.Series(index=df.index, dtype=float)
    atr.iloc[0] = tr.iloc[0]
    for i in range(1, len(df)):
        atr.iloc[i] = atr.iloc[i-1] + (tr.iloc[i] - atr.iloc[i-1]) / 200.0
    
    # Shifted Series for bull signal
    low_3 = low.shift(3)
    high_1 = high.shift(1)
    close_2 = close.shift(2)
    
    # Bullish Signal: bull = showBull and low[3] > high[1] and close[2] < low[3] and close > low[3] and filter(low[3], high[1])
    bull = pd.Series(True, index=df.index)
    bull = bull & showBull
    bull = bull & (low_3 > high_1)
    bull = bull & (close_2 < low_3)
    bull = bull & (close > low_3)
    bull = bull & ((low_3 - high_1) > atr * filterWidth)
    
    # Shifted Series for bear signal
    low_1 = low.shift(1)
    high_3 = high.shift(3)
    close_2_bear = close.shift(2)
    
    # Bearish Signal: bear = showBear and low[1] > high[3] and close[2] > high[3] and close < high[3] and filter(low[1], high[3])
    bear = pd.Series(True, index=df.index)
    bear = bear & showBear
    bear = bear & (low_1 > high_3)
    bear = bear & (close_2_bear > high_3)
    bear = bear & (close < high_3)
    bear = bear & ((low_1 - high_3) > atr * filterWidth)
    
    # Get indices where signals occur
    bull_indices = df.index[bull].tolist()
    bear_indices = df.index[bear].tolist()
    
    entries = []
    trade_num = 1
    
    # Process bullish signals (long entries)
    for i in bull_indices:
        ts = int(df['time'].iloc[i])
        entry_price = float(df['close'].iloc[i])
        entry = {
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
        }
        entries.append(entry)
        trade_num += 1
    
    # Process bearish signals (short entries)
    for i in bear_indices:
        ts = int(df['time'].iloc[i])
        entry_price = float(df['close'].iloc[i])
        entry = {
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
        }
        entries.append(entry)
        trade_num += 1
    
    return entries