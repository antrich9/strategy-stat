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
    
    df = df.copy()
    df['time_dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Resample to 4H
    df_4h = df.set_index('time_dt').resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Detect new 4H candles
    df_4h['is_new_4h'] = True
    
    # 4H data
    low_4h = df_4h['low']
    high_4h = df_4h['high']
    close_4h = df_4h['close']
    volume_4h = df_4h['volume']
    
    # Volume filter
    vol_sma_4h = volume_4h.rolling(9).mean()
    volfilt1 = volume_4h.shift(1) > vol_sma_4h.shift(1) * 1.5
    
    # ATR filter (Wilder)
    tr_4h = np.maximum(
        high_4h - low_4h,
        np.maximum(
            np.abs(high_4h - close_4h.shift(1)),
            np.abs(low_4h - close_4h.shift(1))
        )
    )
    atr_4h = tr_4h.ewm(alpha=1/20, adjust=False).mean() / 1.5
    atrfilt1 = (low_4h - high_4h.shift(2) > atr_4h) | (low_4h.shift(2) - high_4h > atr_4h)
    
    # Trend filter
    loc1 = close_4h.rolling(54).mean()
    locfiltb1 = loc1 > loc1.shift(1)
    locfilts1 = ~locfiltb1
    
    # Bullish and Bearish FVG
    bfvg1 = (low_4h > high_4h.shift(2)) & volfilt1 & atrfilt1 & locfiltb1
    sfvg1 = (high_4h < low_4h.shift(2)) & volfilt1 & atrfilt1 & locfilts1
    
    # Track last FVG type
    lastFVG = pd.Series(0, index=df_4h.index)
    
    # Get confirmed bars from original df
    # For simplicity, we process all bars but entry at confirmed 4H candle start
    confirmed_indices = set()
    
    for i in range(1, len(df)):
        curr_ts = df['time_dt'].iloc[i]
        prev_ts = df['time_dt'].iloc[i-1]
        curr_4h = curr_ts.floor('4h')
        prev_4h = prev_ts.floor('4h')
        if curr_4h != prev_4h:
            confirmed_indices.add(i)
    
    confirmed_indices = sorted(confirmed_indices)
    
    for idx in confirmed_indices:
        row = df.iloc[idx]
        ts = int(row['time'])
        
        curr_4h_ts = row['time_dt'].floor('4h')
        
        if curr_4h_ts in bfvg1.index:
            bfvg_val = bfvg1.loc[curr_4h_ts]
            sfvg_val = sfvg1.loc[curr_4h_ts]
            last_fvg_val = lastFVG.loc[curr_4h_ts] if curr_4h_ts in lastFVG.index else 0
            
            if bfvg_val and last_fvg_val == -1:
                trade_num += 1
                entry_price = float(row['close'])
                results.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
            elif sfvg_val and last_fvg_val == 1:
                trade_num += 1
                entry_price = float(row['close'])
                results.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
            
            # Update lastFVG
            if curr_4h_ts in lastFVG.index:
                if bfvg_val:
                    lastFVG.loc[curr_4h_ts] = 1
                elif sfvg_val:
                    lastFVG.loc[curr_4h_ts] = -1
    
    return results