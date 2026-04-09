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
    high = df['high']
    low = df['low']
    
    # Hull MA parameters
    length_hull = 9
    src_hull = close
    
    # WMA implementation
    def wma(series, length):
        weights = np.arange(1, length + 1)
        def weighted_avg(x):
            return np.dot(x, weights) / weights.sum()
        return series.rolling(window=length, min_periods=length).apply(weighted_avg, raw=True)
    
    # Hull MA calculation
    half_len = int(length_hull / 2)
    sqrt_len = int(np.sqrt(length_hull))
    
    hull_ma = wma(2 * wma(src_hull, half_len) - wma(src_hull, length_hull), sqrt_len)
    
    # Hull MA color/signal: 1 if rising, -1 if falling
    hull_rising = hull_ma > hull_ma.shift(1)
    
    # Long condition: hull rising AND close > hull_ma
    long_cond = hull_rising & (close > hull_ma)
    
    # Short condition: hull falling AND close < hull_ma
    short_cond = (~hull_rising) & (close < hull_ma)
    
    # Iterate and generate entries (only when flat)
    entries = []
    trade_num = 1
    in_position = False
    
    for i in range(len(df)):
        # Skip if indicator is NaN
        if pd.isna(hull_ma.iloc[i]):
            continue
        
        # Long entry
        if long_cond.iloc[i] and not in_position:
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
            in_position = True
        
        # Short entry
        if short_cond.iloc[i] and not in_position:
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
            in_position = True
        
        # Reset when flat (Pine: strategy.position_size == 0)
        # Reset on next bar when not in a position
        if not long_cond.iloc[i] and not short_cond.iloc[i]:
            in_position = False
    
    return entries