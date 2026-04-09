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
    close = df['close']
    high = df['high']
    low = df['low']
    open_price = df['open']
    volume = df['volume']
    
    # Time filter
    timestamps = df['time'].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc))
    hours = timestamps.apply(lambda x: x.hour)
    isValidTradeTime = ((hours >= 2) & (hours < 5)) | ((hours >= 10) & (hours < 12))
    
    # Helper series
    isUp = close > open_price
    isDown = close < open_price
    
    # ATR (Wilder)
    tr = np.maximum(high - low, np.maximum(np.abs(high - close.shift(1)), np.abs(low - close.shift(1))))
    atr_raw = tr.ewm(alpha=1/20, adjust=False).mean()
    atr = atr_raw / 1.5
    
    # Volume filter
    vol_sma = volume.rolling(9).mean()
    volfilt = volume.shift(1) > vol_sma * 1.5
    
    # ATR filter
    atrfilt = ((low - high.shift(2) > atr) | (low.shift(2) - high > atr))
    
    # Trend filter
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    # FVG conditions
    bfvg = (low > high.shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (high < low.shift(2)) & volfilt & atrfilt & locfilts
    
    # OB conditions
    obUp = isDown.shift(1) & isUp & (close > high.shift(1))
    obDown = isUp.shift(1) & isDown & (close < low.shift(1))
    
    # FVG conditions (shifted)
    fvgUp = low.shift(-1) > high.shift(-3)
    fvgDown = high.shift(-1) < low.shift(-3)
    
    # Entry conditions (using current bar index i for ob/fvg checks)
    long_cond = obUp & fvgUp & isValidTradeTime
    short_cond = obDown & fvgDown & isValidTradeTime
    
    entries = []
    trade_num = 1
    
    for i in range(1, len(df)):
        direction = None
        if long_cond.iloc[i]:
            direction = 'long'
        elif short_cond.iloc[i]:
            direction = 'short'
        
        if direction:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': ts,
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