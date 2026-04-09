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
    
    open_vals = df['open']
    high_vals = df['high']
    low_vals = df['low']
    close_vals = df['close']
    
    # Fair Value Gaps - Bullish
    bull_gap_top = np.minimum(close_vals, open_vals)
    bull_gap_btm = np.maximum(close_vals.shift(1), open_vals.shift(1))
    
    bull_fvg = (
        (low_vals > high_vals.shift(2)) &
        (close_vals.shift(1) > high_vals.shift(2)) &
        (open_vals.shift(2) < close_vals.shift(2)) &
        (open_vals.shift(1) < close_vals.shift(1)) &
        (open_vals < close_vals)
    )
    
    # Fair Value Gaps - Bearish
    bear_gap_top = np.minimum(close_vals.shift(1), open_vals.shift(1))
    bear_gap_btm = np.maximum(close_vals, open_vals)
    
    bear_fvg = (
        (high_vals < low_vals.shift(2)) &
        (close_vals.shift(1) < low_vals.shift(2)) &
        (open_vals.shift(2) > close_vals.shift(2)) &
        (open_vals.shift(1) > close_vals.shift(1)) &
        (open_vals > close_vals)
    )
    
    # Opening Gaps - Bullish
    bull_og = low_vals > high_vals.shift(1)
    
    # Opening Gaps - Bearish
    bear_og = high_vals < low_vals.shift(1)
    
    # Volume Imbalances - Bullish
    bull_vi = (
        (open_vals > close_vals.shift(1)) &
        (high_vals.shift(1) > low_vals) &
        (close_vals > close_vals.shift(1)) &
        (open_vals > open_vals.shift(1)) &
        (high_vals.shift(1) < bull_gap_top)
    )
    
    # Volume Imbalances - Bearish
    bear_vi = (
        (open_vals < close_vals.shift(1)) &
        (low_vals.shift(1) < high_vals) &
        (close_vals < close_vals.shift(1)) &
        (open_vals < open_vals.shift(1)) &
        (low_vals.shift(1) > bear_gap_btm)
    )
    
    # Combine all bullish signals
    long_signal = bull_fvg | bull_og | bull_vi
    long_signal = long_signal.fillna(False)
    
    # Combine all bearish signals
    short_signal = bear_fvg | bear_og | bear_vi
    short_signal = short_signal.fillna(False)
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        entry_price = close_vals.iloc[i]
        ts = df['time'].iloc[i]
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        
        if long_signal.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        
        if short_signal.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries