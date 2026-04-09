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
    open_ = df['open']
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    time_col = df['time']
    
    n = len(df)
    
    ob_up = pd.Series(False, index=df.index)
    ob_down = pd.Series(False, index=df.index)
    
    for i in range(1, n):
        if close.iloc[i] > open_.iloc[i] and close.iloc[i-1] < open_.iloc[i-1] and close.iloc[i] > high.iloc[i-1]:
            ob_up.iloc[i] = True
    
    for i in range(1, n):
        if close.iloc[i] < open_.iloc[i] and close.iloc[i-1] > open_.iloc[i-1] and close.iloc[i] < low.iloc[i-1]:
            ob_down.iloc[i] = True
    
    fvg_up = low > high.shift(2)
    fvg_down = high < low.shift(2)
    
    vol_filt = volume.shift(1) > volume.rolling(9).mean() * 1.5
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean()
    atrfilt_val = atr / 1.5
    
    cond1_long = low - high.shift(2) > atrfilt_val
    cond1_short = low.shift(2) - high > atrfilt_val
    atrfilt_long = cond1_long
    atrfilt_short = cond1_short
    
    sma_54 = close.rolling(54).mean()
    loc2 = sma_54 > sma_54.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    bfvg = fvg_up & vol_filt & atrfilt_long & locfiltb
    sfvg = fvg_down & vol_filt & atrfilt_short & locfilts
    
    dt = pd.to_datetime(time_col, unit='s', utc=True)
    hour = dt.dt.hour
    minute = dt.dt.minute
    
    in_morning = ((hour == 8) | ((hour == 9) & (minute <= 45)))
    in_afternoon = ((hour == 15) | ((hour == 16) & (minute <= 45)))
    in_trading_window = in_morning | in_afternoon
    
    long_cond = ob_up.shift(1) & fvg_up & in_trading_window & bfvg
    short_cond = ob_down.shift(1) & fvg_down & in_trading_window & sfvg
    
    entries = []
    trade_num = 1
    
    for i in range(1, n):
        if long_cond.iloc[i]:
            ts = int(time_col.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
        elif short_cond.iloc[i]:
            ts = int(time_col.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
    
    return entries