import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    if len(df) < 3:
        return []
    
    df = df.copy().sort_values('time').reset_index(drop=True)
    
    # Create 4H period identifier
    df['ts_4h'] = df['time'] // (4 * 3600)
    
    # Aggregate to 4H OHLCV
    df_4h = df.groupby('ts_4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'time': 'first'
    }).reset_index(drop=True)
    
    # Volume Filter on 4H (volume[1] > sma(volume, 9) * 1.5)
    vol_4h = df_4h['volume']
    sma_vol_4h = vol_4h.rolling(9).mean()
    volfilt1_series = (vol_4h.shift(1) > sma_vol_4h * 1.5)
    
    # ATR Filter on 4H (atr(20) / 1.5)
    high_4h = df_4h['high']
    low_4h = df_4h['low']
    close_4h = df_4h['close']
    
    high_low = high_4h - low_4h
    high_close = (high_4h - close_4h.shift()).abs()
    low_close = (low_4h - close_4h.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_4h = tr.rolling(20).mean()
    atrfilt1_series = ((low_4h - high_4h.shift(2) > atr_4h) | 
                       (low_4h.shift(2) - high_4h > atr_4h))
    
    # Trend Filter on 4H (sma(close, 54) > sma(close, 54)[1])
    sma_close_54 = close_4h.rolling(54).mean()
    loc2 = sma_close_54 > sma_close_54.shift(1)
    locfiltb1_series = loc2
    locfilts1_series = ~loc2
    
    # Bullish/Bearish FVG on 4H
    bfvg1_series = (low_4h > high_4h.shift(2)) & volfilt1_series & atrfilt1_series & locfiltb1_series
    sfvg1_series = (high_4h < low_4h.shift(2)) & volfilt1_series & atrfilt1_series & locfilts1_series
    
    # Sharp turn conditions
    bullish_sharp_turn_4h = bfvg1_series.copy()
    bearish_sharp_turn_4h = sfvg1_series.copy()
    
    # Track last FVG state to detect sharp turns
    lastFVG_arr = np.zeros(len(df_4h), dtype=int)
    for i in range(1, len(df_4h)):
        if i - 2 >= 0 and pd.notna(bfvg1_series.iloc[i-2]) and bfvg1_series.iloc[i-2]:
            lastFVG_arr[i] = 1
        elif i - 2 >= 0 and pd.notna(sfvg1_series.iloc[i-2]) and sfvg1_series.iloc[i-2]:
            lastFVG_arr[i] = -1
    
    for i in range(2, len(df_4h)):
        if pd.notna(bfvg1_series.iloc[i]) and bfvg1_series.iloc[i] and lastFVG_arr[i-1] == -1:
            bullish_sharp_turn_4h.iloc[i] = True
        if pd.notna(sfvg1_series.iloc[i]) and sfvg1_series.iloc[i] and lastFVG_arr[i-1] == 1:
            bearish_sharp_turn_4h.iloc[i] = True
    
    df_4h['bullish_sharp_turn'] = bullish_sharp_turn_4h
    df_4h['bearish_sharp_turn'] = bearish_sharp_turn_4h
    
    # Merge 4H sharp turn signals back to 15min df
    df = df.merge(df_4h[['ts_4h', 'bullish_sharp_turn', 'bearish_sharp_turn']], on='ts_4h', how='left')
    
    # Identify new 4H candle (first 15min bar of each 4H period)
    is_new_4h = df['ts_4h'] != df['ts_4h'].shift(1)
    
    entries = []
    trade_num = 1
    
    for i in range(1, len(df)):
        if is_new_4h.iloc[i] and pd.notna(df['bullish_sharp_turn'].iloc[i]) and df['bullish_sharp_turn'].iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif is_new_4h.iloc[i] and pd.notna(df['bearish_sharp_turn'].iloc[i]) and df['bearish_sharp_turn'].iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    
    return entries