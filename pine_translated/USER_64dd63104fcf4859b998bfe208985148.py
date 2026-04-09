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
    
    # Resample to 4H bars (assuming input is 15m or lower timeframe)
    # Group by 4-hour periods (14400 seconds)
    df = df.copy()
    if 'time' not in df.columns:
        return []
    
    ts = df['time']
    
    # Check if timestamps are in seconds or milliseconds
    ts_range = ts.max() - ts.min()
    if ts_range > 1e10:  # milliseconds
        divisor = 14400000
    else:  # seconds
        divisor = 14400
    
    df['_4h_group'] = ts // divisor
    
    # Resample to 4H OHLCV
    resampled = df.groupby('_4h_group').agg({
        'time': 'first',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index(drop=True)
    
    if len(resampled) < 3:
        return []
    
    # Calculate 4H ATR (Wilder's method)
    def calc_wilder_atr(high, low, close, length=20):
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        tr.iloc[0] = high.iloc[0] - low.iloc[0]
        
        if len(tr) < length:
            return pd.Series([np.nan] * len(tr), index=tr.index)
        
        atr = pd.Series(np.nan, index=tr.index)
        atr.iloc[length] = tr.iloc[1:length+1].sum()
        
        for i in range(length + 1, len(tr)):
            atr.iloc[i] = (atr.iloc[i-1] * (length - 1) + tr.iloc[i]) / length
        
        return atr
    
    atr_4h = calc_wilder_atr(
        resampled['high'], 
        resampled['low'], 
        resampled['close'], 
        20
    ) / 1.5
    
    # Calculate 4H indicators
    vol_sma_4h = resampled['volume'].rolling(9).mean()
    trend_sma_4h = resampled['close'].rolling(54).mean()
    
    # Extract arrays for faster access
    high_arr = resampled['high'].values
    low_arr = resampled['low'].values
    close_arr = resampled['close'].values
    vol_arr = resampled['volume'].values
    vol_sma_arr = vol_sma_4h.values
    atr_arr = atr_4h.values
    trend_sma_arr = trend_sma_4h.values
    time_arr = resampled['time'].values
    
    entries = []
    trade_num = 1
    lastFVG = 0
    
    n = len(resampled)
    
    for i in range(2, n):
        vol_filt = vol_arr[i-1] > vol_sma_arr[9] * 1.5 if not np.isnan(vol_sma_arr[9]) else True
        
        if not np.isnan(atr_arr[i]):
            atr_filt = (low_arr[i] - high_arr[i-2] > atr_arr[i]) or (low_arr[i-2] - high_arr[i] > atr_arr[i])
        else:
            atr_filt = True
        
        if not np.isnan(trend_sma_arr[i]) and not np.isnan(trend_sma_arr[i-1]):
            trend_bull = trend_sma_arr[i] > trend_sma_arr[i-1]
            trend_bear = not trend_bull
        else:
            trend_bull = True
            trend_bear = True
        
        bfvg = low_arr[i] > high_arr[i-2] and vol_filt and atr_filt and trend_bull
        sfvg = high_arr[i] < low_arr[i-2] and vol_filt and atr_filt and trend_bear
        
        is_new_4h = i == 2 or (time_arr[i] // divisor) != (time_arr[i-1] // divisor)
        
        if is_new_4h:
            if bfvg and lastFVG == -1:
                entry_ts = int(time_arr[i])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': entry_ts,
                    'entry_time': entry_time,
                    'entry_price_guess': close_arr[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close_arr[i],
                    'raw_price_b': close_arr[i]
                })
                trade_num += 1
                lastFVG = 1
            elif sfvg and lastFVG == 1:
                entry_ts = int(time_arr[i])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': entry_ts,
                    'entry_time': entry_time,
                    'entry_price_guess': close_arr[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close_arr[i],
                    'raw_price_b': close_arr[i]
                })
                trade_num += 1
                lastFVG = -1
            elif bfvg:
                lastFVG = 1
            elif sfvg:
                lastFVG = -1
    
    return entries