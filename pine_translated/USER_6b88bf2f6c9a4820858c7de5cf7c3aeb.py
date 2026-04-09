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
    trade_num = 0
    
    # Prepare data
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Resample to 4H timeframe
    df = df.set_index('time')
    df_4h = df.resample('4H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()
    
    if len(df_4h) < 20:
        return results
    
    # Calculate 4H SMA for trend (54 period as in original)
    sma_54 = df_4h['close'].rolling(54).mean()
    locfiltb = sma_54 > sma_54.shift(1)  # Bullish trend
    locfilts = sma_54 < sma_54.shift(1)  # Bearish trend
    
    # Calculate 4H ATR (20 period Wilder)
    high = df_4h['high'].values
    low = df_4h['low'].values
    close = df_4h['close'].values
    
    tr = np.maximum(high[1:] - low[1:], np.maximum(
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1])
    ))
    tr = np.concatenate([[np.nan], tr])
    
    atr = np.zeros(len(df_4h))
    atr[0] = np.nan
    alpha = 1.0 / 20.0
    for i in range(1, len(df_4h)):
        if i == 19:
            atr[i] = np.mean(tr[:20])
        elif i > 19:
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
    
    atr_series = pd.Series(atr, index=df_4h.index)
    
    # Calculate 4H volume SMA (9 period)
    vol_sma_9 = df_4h['volume'].rolling(9).mean()
    
    # Detect Bullish FVG: low > high[2] (gap up)
    bfvg1 = (df_4h['low'].values > np.roll(df_4h['high'].values, 2))
    bfvg1[0:2] = False
    bfvg1 = pd.Series(bfvg1, index=df_4h.index)
    
    # Detect Bearish FVG: high < low[2] (gap down)
    sfvg1 = (df_4h['high'].values < np.roll(df_4h['low'].values, 2))
    sfvg1[0:2] = False
    sfvg1 = pd.Series(sfvg1, index=df_4h.index)
    
    # Apply filters
    volfilt1 = df_4h['volume'].shift(1) > vol_sma_9 * 1.5
    atrfilt1 = ((df_4h['low'] - df_4h['high'].shift(2) > atr_series / 1.5) | 
                (df_4h['low'].shift(2) - df_4h['high'] > atr_series / 1.5))
    
    # Final FVG signals
    bull_fvg = bfvg1 & volfilt1 & atrfilt1 & locfiltb
    bear_fvg = sfvg1 & volfilt1 & atrfilt1 & locfilts
    
    # Sharp turn detection: track lastFVG state
    lastFVG = 0  # 0=None, 1=Bullish, -1=Bearish
    
    for i in range(2, len(df_4h)):
        if pd.isna(bull_fvg.iloc[i]) or pd.isna(bear_fvg.iloc[i]):
            continue
        
        current_bull = bull_fvg.iloc[i]
        current_bear = bear_fvg.iloc[i]
        
        if current_bull and lastFVG == -1:
            # Bullish Sharp Turn entry
            trade_num += 1
            ts = int(df_4h['time'].iloc[i].timestamp())
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': df_4h['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df_4h['close'].iloc[i],
                'raw_price_b': df_4h['close'].iloc[i]
            })
            lastFVG = 1
            
        elif current_bear and lastFVG == 1:
            # Bearish Sharp Turn entry
            trade_num += 1
            ts = int(df_4h['time'].iloc[i].timestamp())
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': df_4h['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df_4h['close'].iloc[i],
                'raw_price_b': df_4h['close'].iloc[i]
            })
            lastFVG = -1
            
        elif current_bull:
            lastFVG = 1
        elif current_bear:
            lastFVG = -1
    
    return results