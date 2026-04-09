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
    
    results = []
    trade_num = 1
    
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    # Filter flags (defaulting to False like in Pine Script)
    inp1 = False  # Volume Filter
    inp2 = False  # ATR Filter
    inp3 = False  # Trend Filter
    
    # Calculate ATR using Wilder's method
    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr1 = true_range.ewm(alpha=1/14, adjust=False).mean()
    atr2 = true_range.ewm(alpha=1/20, adjust=False).mean() / 1.5
    
    # Volume filter
    sma_9_vol = volume.rolling(9).mean()
    volfilt = (volume.shift(1) > sma_9_vol * 1.5) if inp1 else pd.Series(True, index=df.index)
    
    # ATR filter
    atrfilt = (((low - high.shift(2)) > atr2) | ((low.shift(2) - high) > atr2)) if inp2 else pd.Series(True, index=df.index)
    
    # Trend filter
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2 if inp3 else pd.Series(True, index=df.index)
    locfilts = ~loc2 if inp3 else pd.Series(True, index=df.index)
    
    # Bullish FVG condition: low > high[2] (bar index shifted by 2)
    bfvg = (low > high.shift(2)) & volfilt & atrfilt & locfiltb
    
    # Bearish FVG condition: high < low[2]
    sfvg = (high < low.shift(2)) & volfilt & atrfilt & locfilts
    
    for i in range(len(df)):
        if i < 2:
            continue
            
        if pd.isna(bfvg.iloc[i]) or pd.isna(sfvg.iloc[i]):
            continue
            
        if bfvg.iloc[i]:
            entry_price = close.iloc[i]
            entry_ts = df['time'].iloc[i]
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            
        if sfvg.iloc[i]:
            entry_price = close.iloc[i]
            entry_ts = df['time'].iloc[i]
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return results