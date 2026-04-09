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
    time_col = df['time']
    
    ema_length = 50
    fib_level = 0.5
    
    ema = close.ewm(span=ema_length, adjust=False).mean()
    
    long_trend = close > ema
    short_trend = close < ema
    
    window = 5
    swing_high_match = high == high.rolling(window, min_periods=window).max()
    swing_low_match = low == low.rolling(window, min_periods=window).min()
    
    pullback_long_series = pd.Series(np.nan, index=df.index)
    pullback_short_series = pd.Series(np.nan, index=df.index)
    
    last_swing_high = np.nan
    last_swing_low = np.nan
    
    for i in df.index:
        if swing_high_match.loc[i]:
            last_swing_high = high.loc[i]
        if swing_low_match.loc[i]:
            last_swing_low = low.loc[i]
        
        if not np.isnan(last_swing_high) and not np.isnan(last_swing_low):
            pullback_long_series.loc[i] = last_swing_low + fib_level * (last_swing_high - last_swing_low)
            pullback_short_series.loc[i] = last_swing_high - fib_level * (last_swing_high - last_swing_low)
    
    long_entry = long_trend & (close > pullback_long_series) & pullback_long_series.notna()
    short_entry = short_trend & (close < pullback_short_series) & pullback_short_series.notna()
    
    entries = []
    trade_num = 1
    
    for i in df.index:
        ts = time_col.loc[i]
        price = close.loc[i]
        
        if long_entry.loc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(price),
                'raw_price_b': float(price)
            })
            trade_num += 1
        
        if short_entry.loc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(price),
                'raw_price_b': float(price)
            })
            trade_num += 1
    
    return entries