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
    required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    open_px = df['open']
    high_px = df['high']
    low_px = df['low']
    close_px = df['close']
    volume_px = df['volume']
    
    # Volume filter: volume[1] > ta.sma(volume, 9)*1.5
    vol_sma = volume_px.rolling(9).mean()
    vol_filt = volume_px.shift(1) > vol_sma * 1.5
    
    # ATR (Wilder) = ta.atr(20) / 1.5
    high_low = high_px - low_px
    high_close = (high_px - close_px.shift(1)).abs()
    low_close = (low_px - close_px.shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_raw = tr.ewm(alpha=1/20, adjust=False).mean()
    atr = atr_raw / 1.5
    
    # ATR filter: ((low - high[2] > atr) or (low[2] - high > atr))
    atrfilt = ((low_px - high_px.shift(2)) > atr) | ((low_px.shift(2) - high_px) > atr)
    
    # Location filter: ta.sma(close, 54) > loc[1]
    loc = close_px.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    # OB conditions
    is_down = close_px < open_px
    is_up = close_px > open_px
    ob_up = is_down.shift(1) & is_up & (close_px > high_px.shift(1))
    ob_down = is_up.shift(1) & is_down & (close_px < low_px.shift(1))
    
    # FVG conditions
    fvg_up = low_px > high_px.shift(2)
    fvg_down = high_px < low_px.shift(2)
    
    # FVG with filters
    bfvg = fvg_up & vol_filt & atrfilt & locfiltb
    sfvg = fvg_down & vol_filt & atrfilt & locfilts
    
    # Trading window (UTC+1 with DST)
    dt = pd.to_datetime(df['time'], unit='ms', utc=True)
    hours = dt.dt.hour
    minutes = dt.dt.minute
    month = dt.dt.month
    day = dt.dt.day
    dayofweek = dt.dt.dayofweek
    
    # DST: UTC+2 in summer (Mar-Oct), UTC+1 in winter
    is_dst = ((month > 3) & (month < 10)) | \
             ((month == 3) & (((dayofweek == 6) & (day >= 25)) | (day > 25))) | \
             ((month == 10) & (((dayofweek == 6) & (day < 25)) | (day < 25)))
    
    # Adjust hours for UTC+1/UTC+2
    hours_adj = hours + 1 + is_dst.astype(int)
    
    # Window 1: 7:00-10:59, Window 2: 15:00-16:59
    in_window_1 = ((hours_adj == 7) | ((hours_adj >= 8) & (hours_adj <= 9)) | ((hours_adj == 10) & (minutes <= 59)))
    in_window_2 = ((hours_adj == 15) | ((hours_adj == 16) & (minutes <= 59)))
    in_window = in_window_1 | in_window_2
    
    # Entry conditions
    bull_entry = ob_up & fvg_up & in_window
    bear_entry = ob_down & fvg_down & in_window
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if bull_entry.iloc[i] and not pd.isna(close_px.iloc[i]):
            entry_price = close_px.iloc[i]
            ts = int(df['time'].iloc[i])
            dt_entry = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': dt_entry.isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif bear_entry.iloc[i] and not pd.isna(close_px.iloc[i]):
            entry_price = close_px.iloc[i]
            ts = int(df['time'].iloc[i])
            dt_entry = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': dt_entry.isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries