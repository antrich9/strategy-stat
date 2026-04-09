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
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    time_col = df['time']
    
    # inp1, inp2, inp3 - assuming True based on context (not specified in inputs)
    inp1 = True  # Volume Filter
    inp2 = True  # ATR Filter
    inp3 = True  # Trend Filter
    
    # Volume Filter: volume[1] > ta.sma(volume, 9) * 1.5
    vol_sma = volume.shift(1).rolling(9).mean()
    volfilt = volume.shift(1) > vol_sma * 1.5
    
    # ATR Filter: ((low - high[2] > atr) or (low[2] - high > atr))
    # ta.atr(20) / 1.5
    tr1 = high.shift(2) - low
    tr2 = high - low.shift(2)
    tr = pd.concat([tr1, tr2], axis=1).max(axis=1)
    atr = tr.rolling(20).mean()
    atrfilt_val = (low - high.shift(2) > atr / 1.5) | (low.shift(2) - high > atr / 1.5)
    atrfilt = atrfilt_val
    
    # Trend Filter: loc = ta.sma(close, 54), loc2 = loc > loc[1]
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    # Identify Bullish and Bearish FVGs
    # bfvg = low > high[2] and volfilt and atrfilt and locfiltb
    # sfvg = high < low[2] and volfilt and atrfilt and locfilts
    bfvg = (low > high.shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (high < low.shift(2)) & volfilt & atrfilt & locfilts
    
    # Track lastFVG: 1 = Bullish, -1 = Bearish, 0 = None
    lastFVG = pd.Series(0, index=df.index)
    
    # Calculate bullish_entry and bearish_entry
    # bullish_entry = bfvg and lastFVG == -1 and tf > ta.sma(tf, 50)
    # bearish_entry = sfvg and lastFVG == 1 and tf < ta.sma(tf, 50)
    
    # For tf (240 timeframe), we'll use a simple approximation using 240-unit rolling max of time
    # Actually, request.security for 240 is complex - we'll approximate as every 240 bars or use resampling
    # Since we don't have actual 240 TF data, we'll create a proxy using a rolling mean on close
    # The tf check: tf > ta.sma(tf, 50) - this is comparing 240 close to its 50 SMA
    # We'll create tf_proxy as rolling mean to approximate the 240 TF
    tf_proxy = close.rolling(240).mean()  # proxy for 240 TF
    tf_sma50 = tf_proxy.rolling(50).mean()
    
    bullish_entry = bfvg & (lastFVG.shift(1) == -1) & (tf_proxy > tf_sma50)
    bearish_entry = sfvg & (lastFVG.shift(1) == 1) & (tf_proxy < tf_sma50)
    
    # Iterate through bars
    for i in range(len(df)):
        if i < 5:  # Skip bars where required indicators are NaN (swing detection uses [4])
            continue
        
        if bullish_entry.iloc[i]:
            entry_ts = int(time_col.iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])
            
            results.append({
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
        
        if bearish_entry.iloc[i]:
            entry_ts = int(time_col.iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])
            
            results.append({
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
        
        # Update lastFVG tracking
        if bfvg.iloc[i]:
            lastFVG.iloc[i] = 1
        elif sfvg.iloc[i]:
            lastFVG.iloc[i] = -1
        else:
            lastFVG.iloc[i] = lastFVG.iloc[i-1] if i > 0 else 0
    
    return results