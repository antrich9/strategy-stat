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
    if len(df) < 3:
        return []
    
    # Input parameters
    wickToBodyRatio = 0.3
    minBodyPct = 0.6
    inp1 = False  # Volume Filter (from input.bool)
    inp2 = False  # ATR Filter (from input.bool)
    inp3 = False  # Trend Filter (from input.bool)
    atr_length1 = 20
    loc_length = 54
    
    # Create datetime column
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Resample to 4H for indicators
    df_4h = df.set_index('datetime').resample('240T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().copy()
    
    if len(df_4h) < 3:
        return []
    
    high_4h = df_4h['high']
    low_4h = df_4h['low']
    close_4h = df_4h['close']
    volume_4h = df_4h['volume']
    
    # Create mapping from original index to 4H index
    idx_4h = df_4h.index
    df['idx_4h'] = pd.NaT
    for i in range(len(idx_4h) - 1):
        mask = (df['datetime'] >= idx_4h[i]) & (df['datetime'] < idx_4h[i + 1])
        df.loc[mask, 'idx_4h'] = i
    mask_last = df['datetime'] >= idx_4h[-1]
    df.loc[mask_last, 'idx_4h'] = len(idx_4h) - 1
    df['idx_4h'] = df['idx_4h'].astype(int)
    
    # Volume Filter on 4H data
    if inp1:
        vol_sma_4h = volume_4h.rolling(9).mean()
        volfilt_4h = volume_4h.shift(1) > vol_sma_4h * 1.5
    else:
        volfilt_4h = pd.Series(True, index=volume_4h.index)
    
    # Wilder ATR on 4H
    prev_close_4h = close_4h.shift(1)
    tr_4h = pd.concat([high_4h - low_4h, (high_4h - prev_close_4h).abs(), (low_4h - prev_close_4h).abs()], axis=1).max(axis=1)
    atr_4h = tr_4h.ewm(alpha=1.0/atr_length1, adjust=False).mean()
    atr_threshold = atr_4h / 1.5
    
    # ATR Filter on 4H data
    if inp2:
        atrfilt_4h = (low_4h - high_4h.shift(2) > atr_threshold) | (low_4h.shift(2) - high_4h > atr_threshold)
    else:
        atrfilt_4h = pd.Series(True, index=high_4h.index)
    
    # Trend Filter on 4H data (54-period SMA)
    loc_sma = close_4h.rolling(loc_length).mean()
    loc_trend_up = loc_sma > loc_sma.shift(1)
    if inp3:
        locfiltb = loc_trend_up
        locfilts = ~loc_trend_up
    else:
        locfiltb = pd.Series(True, index=loc_trend_up.index)
        locfilts = pd.Series(True, index=loc_trend_up.index)
    
    # FVG detection on 4H (bullish and bearish)
    bfvg_4h = (low_4h > high_4h.shift(2)) & volfilt_4h & atrfilt_4h & locfiltb
    sfvg_4h = (high_4h < low_4h.shift(2)) & volfilt_4h & atrfilt_4h & locfilts
    
    # Map 4H conditions to original index
    idx_map = df['idx_4h'].values
    bfvg = pd.Series(bfvg_4h.values[idx_map], index=df.index)
    sfvg = pd.Series(sfvg_4h.values[idx_map], index=df.index)
    
    # Detect new 4H candle boundaries
    is_new_4h = pd.Series(False, index=df.index)
    unique_4h_indices = df['idx_4h'].unique()
    for idx in range(1, len(unique_4h_indices)):
        curr_4h_idx = unique_4h_indices[idx]
        first_bar_in_4h = df[df['idx_4h'] == curr_4h_idx].index[0]
        is_new_4h.loc[first_bar_in_4h] = True
    
    # Track FVG state across confirmed 4H bars
    lastFVG = 0
    entries = []
    trade_num = 1
    
    for i in range(1, len(df_4h)):
        curr_idx = df_4h.index[i]
        bars_at_this_4h = df[df['idx_4h'] == i]
        if len(bars_at_this_4h) == 0:
            continue
        first_bar_idx = bars_at_this_4h.index[0]
        if not is_new_4h.loc[first_bar_idx]:
            continue
        if bfvg_4h.iloc[i] and lastFVG == -1:
            lastFVG = 1
            ts = int(df_4h['datetime'].iloc[i].timestamp())
            close_price = float(close_4h.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': '',
                'entry_price_guess': close_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close_price,
                'raw_price_b': close_price
            })
            trade_num += 1
        elif sfvg_4h.iloc[i] and lastFVG == 1:
            lastFVG = -1
            ts = int(df_4h['datetime'].iloc[i].timestamp())
            close_price = float(close_4h.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': '',
                'entry_price_guess': close_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close_price,
                'raw_price_b': close_price
            })
            trade_num += 1
        elif bfvg_4h.iloc[i]:
            lastFVG = 1
        elif sfvg_4h.iloc[i]:
            lastFVG = -1
    
    # Map entries to original timeframe
    for entry in entries:
        entry_ts = entry['entry_ts']
        dt_target = pd.Timestamp(entry_ts, unit='s', tz=timezone.utc)
        closest_idx = (df['datetime'] - dt_target).abs().idxmin()
        entry['entry_ts'] = int(df['time'].loc[closest_idx])
        entry['entry_time'] = df['datetime'].loc[closest_idx].isoformat()
        entry['entry_price_guess'] = float(df['close'].loc[closest_idx])
        entry['raw_price_a'] = entry['entry_price_guess']
        entry['raw_price_b'] = entry['entry_price_guess']
    
    entries.sort(key=lambda x: x['entry_ts'])
    for i, entry in enumerate(entries):
        entry['trade_num'] = i + 1
    
    return entries