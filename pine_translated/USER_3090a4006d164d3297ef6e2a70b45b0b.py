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
    
    n = len(df)
    if n < 3:
        return results
    
    time_arr = df['time'].values
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['close'].values
    volume_arr = df['volume'].values
    
    # ATR calculation (Wilder)
    tr = np.maximum(high_arr - low_arr, np.maximum(np.abs(high_arr[1:] - low_arr[:-1]), np.abs(low_arr[1:] - high_arr[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr_20 = np.zeros(n)
    atr_20[0] = np.nan
    atr_20[1] = np.nan
    for i in range(2, n):
        atr_20[i] = (atr_20[i-1] * 19 + tr[i]) / 20
    
    # Volume SMA
    vol_sma_9 = pd.Series(volume_arr).rolling(9).mean().values
    
    # ATR filter (chart timeframe)
    atr2 = atr_20 / 1.5
    atrfilt = (low_arr - np.roll(high_arr, 2) > atr2) | (np.roll(low_arr, 2) - high_arr > atr2)
    atrfilt[0] = False
    atrfilt[1] = False
    
    # Volume filter
    volfilt = volume_arr[1:] > vol_sma_9[1:] * 1.5
    volfilt = np.concatenate([[False], volfilt])
    
    # Trend filter (SMA 54)
    sma_54 = pd.Series(df['close'].values).rolling(54).mean().values
    loc2 = sma_54 > np.roll(sma_54, 1)
    loc2[0] = False
    
    # FVG conditions on chart timeframe
    bfvg = (low_arr > np.roll(high_arr, 2)) & volfilt & atrfilt & loc2
    sfvg = (high_arr < np.roll(low_arr, 2)) & volfilt & atrfilt & (~loc2)
    
    bfvg[0] = False
    bfvg[1] = False
    sfvg[0] = False
    sfvg[1] = False
    
    # Detect new 4H candle
    df['time_dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['4h_floor'] = df['time_dt'].dt.floor('4h')
    is_new_4h = df['4h_floor'].diff().fillna(0) != pd.Timedelta(0)
    is_new_4h.iloc[0] = True
    is_new_4h_arr = is_new_4h.values
    
    # State variables
    lastFVG = 0
    confirmed = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        if i >= 2:
            is_confirmed = True
        else:
            is_confirmed = True
        
        confirmed[i] = is_confirmed
        
        if confirmed[i] and is_new_4h_arr[i]:
            if bfvg[i] and lastFVG == -1:
                trade_num += 1
                ts = int(time_arr[i])
                entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                entry_price = float(df['close'].iloc[i])
                results.append({
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
                lastFVG = 1
            elif sfvg[i] and lastFVG == 1:
                trade_num += 1
                ts = int(time_arr[i])
                entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                entry_price = float(df['close'].iloc[i])
                results.append({
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
                lastFVG = -1
            elif bfvg[i]:
                lastFVG = 1
            elif sfvg[i]:
                lastFVG = -1
    
    return results