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
    
    required = ['time', 'open', 'high', 'low', 'close', 'volume']
    if not all(c in df.columns for c in required):
        raise ValueError(f"df must contain columns: {required}")
    
    df = df.copy()
    df = df.sort_values('time').reset_index(drop=True)
    
    n = 16
    if len(df) < n:
        return []
    
    high_4h = df['high'].rolling(n).max()
    low_4h = df['low'].rolling(n).min()
    close_4h = df['close'].iloc[n-1::n].values
    close_4h = pd.Series(close_4h, index=df.index[n-1::n])
    volume_4h = df['volume'].rolling(n).sum()
    time_4h = df['time'].iloc[n-1::n].values
    
    idx = df.index[n-1::n]
    df_4h = pd.DataFrame({
        'high': high_4h.loc[idx].values,
        'low': low_4h.loc[idx].values,
        'close': close_4h.loc[idx].values,
        'volume': volume_4h.loc[idx].values,
        'time': time_4h
    }, index=idx).reset_index(drop=True)
    
    tr = np.maximum(
        df_4h['high'] - df_4h['low'],
        np.maximum(
            np.abs(df_4h['high'] - df_4h['close'].shift(1)),
            np.abs(df_4h['low'] - df_4h['close'].shift(1))
        )
    )
    
    length = 20
    atr_4h = np.zeros(len(df_4h))
    atr_4h[length-1] = tr.iloc[:length].mean()
    for i in range(length, len(tr)):
        atr_4h[i] = (atr_4h[i-1] * (length - 1) + tr.iloc[i]) / length
    atr_4h = pd.Series(atr_4h, index=df_4h.index)
    atr_4h = atr_4h.replace(0, np.nan)
    
    vol_sma_4h = df_4h['volume'].rolling(9).mean()
    vol_filt = df_4h['volume'] > vol_sma_4h * 1.5
    
    loc1 = df_4h['close'].rolling(54).mean()
    loc21 = loc1 > loc1.shift(1)
    locfiltb = loc21
    locfilts = ~loc21
    
    gap_up = df_4h['low'] - df_4h['high'].shift(2)
    gap_down = df_4h['high'] - df_4h['low'].shift(2)
    atr_filt = (gap_up > atr_4h / 1.5) | (gap_down > atr_4h / 1.5)
    
    bfvg = (df_4h['low'] > df_4h['high'].shift(2)) & vol_filt & atr_filt & locfiltb
    sfvg = (df_4h['high'] < df_4h['low'].shift(2)) & vol_filt & atr_filt & locfilts
    
    entries = []
    last_fvg = 0
    
    for i in range(len(df_4h)):
        if np.isnan(bfvg.iloc[i]) or np.isnan(sfvg.iloc[i]):
            continue
        
        if bfvg.iloc[i]:
            if last_fvg == -1:
                entry_ts = int(df_4h['time'].iloc[i])
                entries.append({
                    'trade_num': len(entries) + 1,
                    'direction': 'long',
                    'entry_ts': entry_ts,
                    'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(df_4h['close'].iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(df_4h['close'].iloc[i]),
                    'raw_price_b': float(df_4h['close'].iloc[i])
                })
            last_fvg = 1
        elif sfvg.iloc[i]:
            if last_fvg == 1:
                entry_ts = int(df_4h['time'].iloc[i])
                entries.append({
                    'trade_num': len(entries) + 1,
                    'direction': 'short',
                    'entry_ts': entry_ts,
                    'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(df_4h['close'].iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(df_4h['close'].iloc[i]),
                    'raw_price_b': float(df_4h['close'].iloc[i])
                })
            last_fvg = -1
    
    return entries