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
    open_arr = df['open']
    high = df['high']
    low = df['low']
    
    ema9_1d = close.ewm(span=9, adjust=False).mean()
    ema18_1d = close.ewm(span=18, adjust=False).mean()
    
    is_up = close > open_arr
    is_down = close < open_arr
    
    is_ob_up = (is_down.shift(1)) & (is_up) & (close > high.shift(1))
    is_ob_down = (is_up.shift(1)) & (is_down) & (close < low.shift(1))
    
    is_fvg_up = low > high.shift(2)
    is_fvg_down = high < low.shift(2)
    
    ob_up = is_ob_up.shift(1)
    ob_down = is_ob_down.shift(1)
    fvg_up = is_fvg_up
    fvg_down = is_fvg_down
    
    bullish_entry = ob_up & fvg_up & (close > ema9_1d) & (close > ema18_1d)
    bearish_entry = ob_down & fvg_down & (close < ema9_1d) & (close < ema18_1d)
    
    entries = []
    trade_num = 1
    
    for i in range(3, len(df)):
        if pd.isna(ema9_1d.iloc[i]) or pd.isna(ema18_1d.iloc[i]):
            continue
        
        if bullish_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
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
        elif bearish_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
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