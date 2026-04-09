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
    
    # Resample to 4H timeframe
    df_4h = df.copy()
    df_4h['time'] = pd.to_datetime(df_4h['time'], unit='s', utc=True)
    df_4h.set_index('time', inplace=True)
    
    # Resample to 4H using OHLCV
    df_4h_resampled = df_4h.resample('4H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Convert time back to unix timestamp
    df_4h_resampled['time'] = df_4h_resampled.index.astype('int64') // 10**9
    df_4h_resampled = df_4h_resampled.reset_index(drop=True)
    
    # Calculate indicators on 4H data
    high_4h = df_4h_resampled['high'].values
    low_4h = df_4h_resampled['low'].values
    close_4h = df_4h_resampled['close'].values
    volume_4h = df_4h_resampled['volume'].values
    
    # Volume Filter
    volume_4h_shifted = np.roll(volume_4h, 1)
    volume_4h_shifted[0] = np.nan
    sma_9 = pd.Series(volume_4h).rolling(9).mean().values
    volfilt1 = volume_4h_shifted > sma_9 * 1.5
    
    # ATR Filter (Wilder ATR)
    tr = np.maximum(high_4h - low_4h, 
                    np.maximum(np.abs(high_4h - np.roll(close_4h, 1)),
                               np.abs(low_4h - np.roll(close_4h, 1))))
    tr[0] = np.nan
    atr_4h = np.zeros_like(tr)
    atr_4h[19] = np.mean(tr[1:20])
    for i in range(20, len(tr)):
        atr_4h[i] = (atr_4h[i-1] * 19 + tr[i]) / 20
    atr_4h[:20] = np.nan
    atr_4h_scaled = atr_4h / 1.5
    
    # FVG conditions - need at least 2 bars of history
    atrfilt1 = np.zeros_like(high_4h, dtype=bool)
    for i in range(2, len(high_4h)):
        atrfilt1[i] = (low_4h[i] - high_4h[i-2] > atr_4h_scaled[i]) or (low_4h[i-2] - high_4h[i] > atr_4h_scaled[i])
    
    # Trend Filter
    loc1 = pd.Series(close_4h).rolling(54).mean().values
    loc21 = loc1 > np.roll(loc1, 1)
    loc21[0] = False
    locfiltb1 = loc21.copy()
    locfilts1 = ~loc21
    
    # Bullish/Bearish FVG Detection
    bfvg1 = np.zeros(len(high_4h), dtype=bool)
    sfvg1 = np.zeros(len(high_4h), dtype=bool)
    
    for i in range(2, len(high_4h)):
        bfvg1[i] = low_4h[i] > high_4h[i-2] and volfilt1[i] and atrfilt1[i] and locfiltb1[i]
        sfvg1[i] = high_4h[i] < low_4h[i-2] and volfilt1[i] and atrfilt1[i] and locfilts1[i]
    
    # Sharp Turn Detection - entries when current FVG type differs from previous
    entries = []
    trade_num = 1
    lastFVG = 0  # 0 = None, 1 = Bullish, -1 = Bearish
    
    for i in range(3, len(bfvg1)):
        # Skip if any required indicator is NaN
        if np.isnan(volfilt1[i]) or np.isnan(atrfilt1[i]) or np.isnan(loc1[i]) or np.isnan(loc1[i-1]):
            continue
        
        if bfvg1[i] and lastFVG == -1:
            # Bullish Sharp Turn - LONG entry
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df_4h_resampled['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df_4h_resampled['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': close_4h[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close_4h[i],
                'raw_price_b': close_4h[i]
            })
            trade_num += 1
            lastFVG = 1
        elif sfvg1[i] and lastFVG == 1:
            # Bearish Sharp Turn - SHORT entry
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df_4h_resampled['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df_4h_resampled['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': close_4h[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close_4h[i],
                'raw_price_b': close_4h[i]
            })
            trade_num += 1
            lastFVG = -1
        elif bfvg1[i]:
            lastFVG = 1
        elif sfvg1[i]:
            lastFVG = -1
    
    return entries