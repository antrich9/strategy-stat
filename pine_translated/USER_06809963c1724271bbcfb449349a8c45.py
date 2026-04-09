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
    # Convert time to datetime for timezone-aware comparison
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Trading windows: 07:00-11:45 and 14:00-14:45 Europe/London
    # Since df is already UTC, we adjust for Europe/London (UTC+0 or UTC+1 for BST)
    def in_trading_window(dt):
        hour = dt.hour
        minute = dt.minute
        time_minutes = hour * 60 + minute
        # Window 1: 07:00-11:45
        w1_start = 7 * 60
        w1_end = 11 * 60 + 45
        # Window 2: 14:00-14:45
        w2_start = 14 * 60
        w2_end = 14 * 60 + 45
        return (w1_start <= time_minutes < w1_end) or (w2_start <= time_minutes < w2_end)
    
    df['in_window'] = df['datetime'].apply(in_trading_window)
    
    # For 4H data, we need to resample to 4H timeframe
    # Convert to period-based approach
    df['4h_period'] = df['time'].apply(lambda x: x // (4 * 3600))
    df_4h = df.groupby('4h_period').agg({
        'time': 'last',
        'datetime': 'last',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index(drop=True)
    
    high_4h = df_4h['high']
    low_4h = df_4h['low']
    close_4h = df_4h['close']
    volume_4h = df_4h['volume']
    
    # Detect new 4H candle
    is_new_4h = np.concatenate([[True], df_4h['time'].values[1:] != df_4h['time'].values[:-1]])
    
    # Volume Filter: volume[1] > SMA(volume, 9) * 1.5
    vol_sma = volume_4h.rolling(9).mean() * 1.5
    volfilt = (volume_4h.shift(1) > vol_sma.shift(1))
    
    # ATR Filter: ATR(20) / 1.5
    atr_length = 20
    tr = np.maximum(high_4h - low_4h, np.maximum(
        np.abs(high_4h - close_4h.shift(1)),
        np.abs(low_4h - close_4h.shift(1))
    ))
    atr = tr.ewm(alpha=1.0/atr_length, adjust=False).mean() / 1.5
    atrfilt = ((low_4h - high_4h.shift(2) > atr.shift(1)) | (low_4h.shift(2) - high_4h.shift(1) > atr.shift(1)))
    
    # Trend Filter: SMA(close, 54) > SMA(close, 54)[1]
    loc = close_4h.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    # Bullish FVG: low > high[2]
    bfvg = (low_4h > high_4h.shift(2)) & volfilt & atrfilt & locfiltb
    # Bearish FVG: high < low[2]
    sfvg = (high_4h < low_4h.shift(2)) & volfilt & atrfilt & locfilts
    
    # Track last FVG type
    lastFVG = 0
    entries = []
    trade_num = 1
    
    for i in range(1, len(df_4h)):
        if not is_new_4h[i]:
            continue
        
        prev_lastFVG = lastFVG
        
        # Detect Sharp Turn entries
        if bfvg.iloc[i] and lastFVG == -1:
            if df_4h['in_window'].iloc[i]:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(df_4h['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df_4h['time'].iloc[i], tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(close_4h.iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(close_4h.iloc[i]),
                    'raw_price_b': float(close_4h.iloc[i])
                })
                trade_num += 1
            lastFVG = 1
        elif sfvg.iloc[i] and lastFVG == 1:
            if df_4h['in_window'].iloc[i]:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(df_4h['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df_4h['time'].iloc[i], tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(close_4h.iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(close_4h.iloc[i]),
                    'raw_price_b': float(close_4h.iloc[i])
                })
                trade_num += 1
            lastFVG = -1
        elif bfvg.iloc[i]:
            lastFVG = 1
        elif sfvg.iloc[i]:
            lastFVG = -1
    
    return entries