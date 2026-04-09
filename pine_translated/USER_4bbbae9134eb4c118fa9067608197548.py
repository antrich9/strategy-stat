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
    
    inp11 = False
    inp21 = False
    inp31 = False
    
    df = df.copy()
    
    df['_4h_start'] = (df['time'] // 14400) * 14400
    
    grouped = df.groupby('_4h_start')
    high_4h = grouped['high'].max()
    low_4h = grouped['low'].min()
    close_4h = grouped['close'].last()
    volume_4h = grouped['volume'].sum()
    
    high_4h.index = high_4h.index.get_level_list(0)
    low_4h.index = low_4h.index.get_level_list(0)
    close_4h.index = close_4h.index.get_level_list(0)
    volume_4h.index = volume_4h.index.get_level_list(0)
    
    high_4h_mapped = df['_4h_start'].map(high_4h).astype(float)
    low_4h_mapped = df['_4h_start'].map(low_4h).astype(float)
    close_4h_mapped = df['_4h_start'].map(close_4h).astype(float)
    volume_4h_mapped = df['_4h_start'].map(volume_4h).astype(float)
    
    is_new_4h = df['_4h_start'].diff().fillna(0) > 0
    
    vol_sma_4h = volume_4h_mapped.rolling(9).mean()
    volfilt1 = volume_4h_mapped.shift(1) > vol_sma_4h.shift(1) * 1.5 if inp11 else pd.Series(True, index=df.index)
    
    atr_length1 = 20
    tr_4h = pd.concat([high_4h_mapped, close_4h_mapped.shift(1)], axis=1).max(axis=1) - pd.concat([low_4h_mapped, close_4h_mapped.shift(1)], axis=1).min(axis=1)
    tr_4h = pd.concat([tr_4h, (high_4h_mapped - low_4h_mapped).abs()], axis=1).max(axis=1)
    atr_4h = tr_4h.ewm(alpha=1/atr_length1, adjust=False).mean() / 1.5
    atrfilt1 = ((low_4h_mapped - high_4h_mapped.shift(2) > atr_4h) | (low_4h_mapped.shift(2) - high_4h_mapped > atr_4h)) if inp21 else pd.Series(True, index=df.index)
    
    loc1 = close_4h_mapped.rolling(54).mean()
    loc21 = loc1 > loc1.shift(1)
    locfiltb1 = loc21 if inp31 else pd.Series(True, index=df.index)
    locfilts1 = (~loc21) if inp31 else pd.Series(True, index=df.index)
    
    bfvg1 = (low_4h_mapped > high_4h_mapped.shift(2)) & volfilt1 & atrfilt1 & locfiltb1
    sfvg1 = (high_4h_mapped < low_4h_mapped.shift(2)) & volfilt1 & atrfilt1 & locfilts1
    
    entries = []
    trade_num = 1
    lastFVG = 0
    
    for i in range(2, len(df)):
        if is_new_4h.iloc[i]:
            prev_bfvg = bfvg1.iloc[i]
            prev_sfvg = sfvg1.iloc[i]
            
            if prev_bfvg and lastFVG == -1:
                ts = df['time'].iloc[i]
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(ts),
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': close_4h_mapped.iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close_4h_mapped.iloc[i],
                    'raw_price_b': close_4h_mapped.iloc[i]
                })
                trade_num += 1
                lastFVG = 1
            elif prev_sfvg and lastFVG == 1:
                ts = df['time'].iloc[i]
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(ts),
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': close_4h_mapped.iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close_4h_mapped.iloc[i],
                    'raw_price_b': close_4h_mapped.iloc[i]
                })
                trade_num += 1
                lastFVG = -1
            elif prev_bfvg:
                lastFVG = 1
            elif prev_sfvg:
                lastFVG = -1
    
    return entries