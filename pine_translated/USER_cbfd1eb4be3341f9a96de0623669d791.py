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
    
    # Volume filter: volume[1] > sma(volume, 9) * 1.5
    volume_sma_9 = df['volume'].rolling(9).mean()
    volfilt = df['volume'].shift(1) > volume_sma_9 * 1.5
    
    # ATR calculation (Wilder)
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(span=20, adjust=False).mean()
    atr_scaled = atr / 1.5
    
    # ATR filter: (low - high[2] > atr) or (low[2] - high > atr)
    atrfilt = (low - high.shift(2) > atr_scaled) | (low.shift(2) - high > atr_scaled)
    
    # Trend filter (loc): sma(close, 54) > sma(close, 54)[1]
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    
    # locfiltb and locfilts
    locfiltb = loc2
    locfilts = ~loc2
    
    # Bullish FVG: low > high[2] and volfilt and atrfilt and locfiltb
    bfvg = (low > high.shift(2)) & volfilt & atrfilt & locfiltb
    
    # Bearish FVG: high < low[2] and volfilt and atrfilt and locfilts
    sfvg = (high < low.shift(2)) & volfilt & atrfilt & locfilts
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if np.isnan(volfilt.iloc[i]) or np.isnan(atrfilt.iloc[i]) or np.isnan(locfiltb.iloc[i]) or np.isnan(locfilts.iloc[i]):
            continue
        
        if bfvg.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
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
        
        if sfvg.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
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