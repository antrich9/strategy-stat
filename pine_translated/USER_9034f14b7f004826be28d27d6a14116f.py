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
    df['ts'] = df['time']
    
    # Resample to 4H using pandas
    df_4h = df.set_index('time').copy()
    df_4h.index = pd.to_datetime(df_4h.index, unit='s', utc=True)
    df_4h_ohlc = df_4h.resample('240T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    df_4h_ohlc = df_4h_ohlc.reset_index()
    df_4h_ohlc = df_4h_ohlc.rename(columns={'time': 'time_4h'})
    df_4h_ohlc['time_4h_int'] = df_4h_ohlc['time_4h'].astype('int64') // 10**9
    
    high_4h = df_4h_ohlc['high'].values
    low_4h = df_4h_ohlc['low'].values
    close_4h = df_4h_ohlc['close'].values
    volume_4h = df_4h_ohlc['volume'].values
    time_4h_int = df_4h_ohlc['time_4h_int'].values
    
    n_4h = len(df_4h_ohlc)
    
    # Volume Filter (4H)
    vol_sma_4h = pd.Series(volume_4h).rolling(9).mean().values
    volfilt1 = volume_4h[1:] > vol_sma_4h[1:] * 1.5
    volfilt1 = np.concatenate([[False], volfilt1])
    
    # ATR Filter (4H) - Wilder ATR
    tr_4h = np.zeros(n_4h)
    for i in range(1, n_4h):
        tr_4h[i] = max(high_4h[i] - low_4h[i], abs(high_4h[i] - close_4h[i-1]), abs(low_4h[i] - close_4h[i-1]))
    atr_len = 20
    atr_4h = np.zeros(n_4h)
    atr_4h[atr_len-1] = np.mean(tr_4h[0:atr_len])
    for i in range(atr_len, n_4h):
        atr_4h[i] = atr_4h[i-1] * (atr_len - 1) / atr_len + tr_4h[i] / atr_len
    atr_4h_adj = atr_4h / 1.5
    atrfilt1 = np.zeros(n_4h, dtype=bool)
    for i in range(2, n_4h):
        atrfilt1[i] = (low_4h[i] - high_4h[i-2] > atr_4h_adj[i]) or (low_4h[i-2] - high_4h[i] > atr_4h_adj[i])
    
    # Trend Filter (4H)
    sma_54_4h = pd.Series(close_4h).rolling(54).mean().values
    loc2_4h = sma_54_4h > np.roll(sma_54_4h, 1)
    loc2_4h[0] = False
    locfiltb1 = loc2_4h.copy()
    locfilts1 = ~loc2_4h
    
    # Bullish/Bearish FVGs (4H)
    bfvg1 = np.zeros(n_4h, dtype=bool)
    sfvg1 = np.zeros(n_4h, dtype=bool)
    for i in range(2, n_4h):
        bfvg1[i] = low_4h[i] > high_4h[i-2] and volfilt1[i] and atrfilt1[i] and locfiltb1[i]
        sfvg1[i] = high_4h[i] < low_4h[i-2] and volfilt1[i] and atrfilt1[i] and locfilts1[i]
    
    # Track last FVG state
    lastFVG = 0
    entries = []
    trade_num = 1
    
    # Detect sharp turns and generate entries
    for i in range(2, n_4h):
        if bfvg1[i] and lastFVG == -1:
            # Bullish Sharp Turn - LONG entry
            entry_ts = int(time_4h_int[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(close_4h[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close_4h[i]),
                'raw_price_b': float(close_4h[i])
            })
            trade_num += 1
            lastFVG = 1
        elif sfvg1[i] and lastFVG == 1:
            # Bearish Sharp Turn - SHORT entry
            entry_ts = int(time_4h_int[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(close_4h[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close_4h[i]),
                'raw_price_b': float(close_4h[i])
            })
            trade_num += 1
            lastFVG = -1
        elif bfvg1[i]:
            lastFVG = 1
        elif sfvg1[i]:
            lastFVG = -1
    
    return entries