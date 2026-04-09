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
    
    # Filter parameters (default values from Pine Script inputs)
    inp1 = False  # Volume Filter
    inp2 = False  # ATR Filter
    inp3 = False  # Trend Filter
    
    # Shifted values for daily data (240H timeframe)
    dailyHigh = df['high']
    dailyLow = df['low']
    dailyHigh1 = df['high'].shift(1)
    dailyLow1 = df['low'].shift(1)
    dailyHigh2 = df['high'].shift(2)
    dailyLow2 = df['low'].shift(2)
    dailyHigh3 = df['high'].shift(3)
    dailyLow3 = df['low'].shift(3)
    dailyHigh4 = df['high'].shift(4)
    dailyLow4 = df['low'].shift(4)
    
    # Volume filter
    if inp1:
        volfilt = df['volume'].shift(1) > df['volume'].rolling(9).mean() * 1.5
    else:
        volfilt = pd.Series(True, index=df.index)
    
    # ATR filter (Wilder ATR)
    if inp2:
        high_low = dailyHigh - dailyLow
        high_close_prev = (dailyHigh - df['close'].shift(1)).abs()
        low_close_prev = (dailyLow - df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0/20, adjust=False).mean()
        atr2 = atr / 1.5
        atrfilt = (dailyLow - dailyHigh2 > atr2) | (dailyLow2 - dailyHigh > atr2)
    else:
        atrfilt = pd.Series(True, index=df.index)
    
    # Trend filter
    if inp3:
        loc = df['close'].rolling(54).mean()
        loc2 = loc > loc.shift(1)
        locfiltb = loc2
        locfilts = ~loc2
    else:
        locfiltb = pd.Series(True, index=df.index)
        locfilts = pd.Series(True, index=df.index)
    
    # Swing detection
    is_swing_high = (dailyHigh1 < dailyHigh2) & (dailyHigh3 < dailyHigh2) & (dailyHigh4 < dailyHigh2)
    is_swing_low = (dailyLow1 > dailyLow2) & (dailyLow3 > dailyLow2) & (dailyLow4 > dailyLow2)
    
    # FVG conditions
    bfvg = (dailyLow > dailyHigh2) & volfilt & atrfilt & locfiltb
    sfvg = (dailyHigh < dailyLow2) & volfilt & atrfilt & locfilts
    
    # Initialize tracking variables
    last_swing_high1 = np.nan
    last_swing_low1 = np.nan
    lastSwingType1 = "none"
    
    entries = []
    trade_num = 1
    
    # Start from index 5 to ensure all indicators are available
    start_idx = 5
    for i in range(start_idx, len(df)):
        # Update swing tracking
        if is_swing_high.iloc[i]:
            last_swing_high1 = dailyHigh2.iloc[i]
            lastSwingType1 = "dailyHigh"
        if is_swing_low.iloc[i]:
            last_swing_low1 = dailyLow2.iloc[i]
            lastSwingType1 = "dailyLow"
        
        # Check entry conditions
        bull_entry = bfvg.iloc[i] and (lastSwingType1 == "dailyLow")
        bear_entry = sfvg.iloc[i] and (lastSwingType1 == "dailyHigh")
        
        if bull_entry:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        
        if bear_entry:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    
    return entries