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
    
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    time = df['time']
    
    # SMA for loc (54)
    loc = close.rolling(54).mean()
    
    # Volume filter
    vol_sma = volume.rolling(9).mean()
    volfilt = volume.shift(1) > vol_sma * 1.5
    
    # ATR (Wilder, period 20)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean()
    atr2 = atr / 1.5
    
    # loc2: loc > loc[1]
    loc2 = loc > loc.shift(1)
    locfiltb = loc2  # bullish trend filter
    locfilts = ~loc2  # bearish trend filter (not loc2)
    
    # FVG components
    dailyHigh2 = high.shift(2)
    dailyLow2 = low.shift(2)
    dailyHigh1 = high.shift(1)
    dailyLow1 = low.shift(1)
    dailyHigh = high
    dailyLow = low
    
    # Bull and Bear FVG conditions
    bfvg = (dailyLow > dailyHigh2) & volfilt & (dailyLow - dailyHigh2 > atr2) & locfiltb
    sfvg = (dailyHigh < dailyLow2) & volfilt & (dailyLow2 - dailyHigh > atr2) & locfilts
    
    # Swing detection
    is_swing_high = (dailyHigh1 < dailyHigh2) & (high.shift(3) < dailyHigh2) & (high.shift(4) < dailyHigh2)
    is_swing_low = (dailyLow1 > dailyLow2) & (low.shift(3) > dailyLow2) & (low.shift(4) > dailyLow2)
    
    # Track last swing type
    lastSwingType = pd.Series(index=df.index, dtype='object')
    current_swing_type = "none"
    
    entries = []
    trade_num = 1
    
    for i in range(1, len(df)):
        # Update swing type
        if is_swing_high.iloc[i]:
            current_swing_type = "dailyHigh"
        if is_swing_low.iloc[i]:
            current_swing_type = "dailyLow"
        
        # Skip if indicators are NaN
        if pd.isna(loc.iloc[i]) or pd.isna(vol_sma.iloc[i]) or pd.isna(atr2.iloc[i]):
            continue
        if pd.isna(dailyHigh2.iloc[i]) or pd.isna(dailyLow2.iloc[i]):
            continue
        
        direction = None
        
        # Long entry: bfvg and lastSwingType == "dailyLow"
        if bfvg.iloc[i] and current_swing_type == "dailyLow":
            direction = "long"
        # Short entry: sfvg and lastSwingType == "dailyHigh"
        elif sfvg.iloc[i] and current_swing_type == "dailyHigh":
            direction = "short"
        
        if direction:
            entry_ts = int(time.iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])
            
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
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