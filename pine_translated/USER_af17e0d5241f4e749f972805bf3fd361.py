import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    df columns: time(int unix ts), open(float), high(float), low(float), close(float), volume(float)
    Rows sorted ascending by time (oldest first). Index is 0-based int.
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Create 4H candles using resample
    df_4h = df.set_index('datetime').resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    high_4h = df_4h['high']
    low_4h = df_4h['low']
    close_4h = df_4h['close']
    volume_4h = df_4h['volume']
    
    # Volume filter (9 SMA)
    vol_sma_4h = volume_4h.rolling(9).mean()
    volfilt1 = volume_4h > vol_sma_4h * 1.5
    
    # ATR filter (Wilder ATR 20, divided by 1.5)
    high_low_4h = high_4h - low_4h
    high_close_prev_4h = np.abs(high_4h - close_4h.shift(1))
    low_close_prev_4h = np.abs(low_4h - close_4h.shift(1))
    tr_4h = pd.concat([high_low_4h, high_close_prev_4h, low_close_prev_4h], axis=1).max(axis=1)
    atr_4h = tr_4h.ewm(alpha=1/20, adjust=False).mean()
    atr_thresh_4h = atr_4h / 1.5
    atrfilt1 = (low_4h - high_4h.shift(2) > atr_thresh_4h) | (low_4h.shift(2) - high_4h > atr_thresh_4h)
    
    # Detect base FVGs on 4H
    bullish_fvg_4h = low_4h > high_4h.shift(2)
    bearish_fvg_4h = high_4h < low_4h.shift(2)
    
    # Align 4H data to chart timeframe
    high_4h_aligned = high_4h.reindex(df.index, method='ffill')
    low_4h_aligned = low_4h.reindex(df.index, method='ffill')
    close_4h_aligned = close_4h.reindex(df.index, method='ffill')
    volfilt1_aligned = volfilt1.reindex(df.index, method='ffill').fillna(False)
    atrfilt1_aligned = atrfilt1.reindex(df.index, method='ffill').fillna(False)
    bullish_fvg_4h_aligned = bullish_fvg_4h.reindex(df.index, method='ffill').fillna(False)
    bearish_fvg_4h_aligned = bearish_fvg_4h.reindex(df.index, method='ffill').fillna(False)
    
    # Trend filter (54 SMA on 4H)
    loc1 = close_4h.rolling(54).mean()
    loc21 = loc1 > loc1.shift(1)
    locfiltb1 = loc21.reindex(df.index, method='ffill').fillna(True)
    locfilts1 = (~loc21).reindex(df.index, method='ffill').fillna(True)
    
    # Apply filters to get final FVGs on 4H
    bfvg1 = bullish_fvg_4h_aligned & volfilt1_aligned & atrfilt1_aligned & locfiltb1
    sfvg1 = bearish_fvg_4h_aligned & volfilt1_aligned & atrfilt1_aligned & locfilts1
    
    # Detect new 4H period
    is_new_4h = np.zeros(len(df), dtype=bool)
    is_new_4h[0] = True
    for i in range(1, len(df)):
        curr_4h_start = df['datetime'].iloc[i].floor('4h')
        prev_4h_start = df['datetime'].iloc[i-1].floor('4h')
        is_new_4h[i] = curr_4h_start != prev_4h_start
    
    # State machine: track FVG