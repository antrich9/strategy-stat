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
    
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']
    volume = df['volume']
    time_col = df['time']
    
    n = len(df)
    entries = []
    trade_num = 1
    
    is_up = close > open_
    is_down = close < open_
    
    ob_up = is_down.shift(1) & is_up & (close > high.shift(1))
    ob_down = is_up.shift(1) & is_down & (close < low.shift(1))
    
    fvg_up = low > high.shift(2)
    fvg_down = high < low.shift(2)
    
    def in_london_window(ts_val):
        dt = datetime.fromtimestamp(ts_val / 1000.0, tz=timezone.utc)
        utc_dt = dt.replace(tzinfo=timezone.utc)
        lon_dt = utc_dt.astimezone(timezone.utc)
        return lon_dt.hour, lon_dt.minute
    
    vol_filt = volume > volume.rolling(9).mean() * 1.5
    
    high_low = high - low
    high_close = (high - close).abs()
    low_close = (low - close).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(20).mean() / 1.5
    atrfilt = (low - high.shift(2) > atr) | (low.shift(2) - high > atr)
    
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    bull_fvg = fvg_up & vol_filt & atrfilt & locfiltb
    bear_fvg = fvg_down & vol_filt & atrfilt & locfilts
    
    bull_entry = bull_fvg & ob_up.shift(1)
    bear_entry = bear_fvg & ob_down.shift(1)
    
    within_window = pd.Series(False, index=df.index)
    for idx in range(n):
        h, m = in_london_window(time_col.iloc[idx])
        time_val = h * 60 + m
        morning = 7 * 60 + 45 <= time_val <= 8 * 60 + 45
        afternoon = 14 * 60 + 45 <= time_val <= 15 * 60 + 45
        within_window.iloc[idx] = morning or afternoon
    
    long_cond = bull_entry & within_window
    short_cond = bear_entry & within_window
    
    direction = pd.Series('', index=df.index)
    direction[long_cond] = 'long'
    direction[short_cond] = 'short'
    
    for i in range(2, n):
        if direction.iloc[i]:
            entry_price_guess = close.iloc[i]
            ts = int(time_col.iloc[i])
            entry_time = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': direction.iloc[i],
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price_guess,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price_guess,
                'raw_price_b': entry_price_guess
            })
            trade_num += 1
    
    return entries