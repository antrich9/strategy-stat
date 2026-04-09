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
    if not all(col in df.columns for col in required_cols):
        return []
    
    if len(df) < 50:
        return []
    
    period = 10
    multiplier = 3.0
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = pd.Series(np.nan, index=df.index)
    atr.iloc[period-1] = tr.iloc[:period].mean()
    multiplier_factor = 1.0 / period
    for i in range(period, len(df)):
        atr.iloc[i] = (atr.iloc[i-1] * (period - 1) + tr.iloc[i]) * multiplier_factor
    
    hl2 = (high + low) / 2
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr
    
    final_upper = pd.Series(np.nan, index=df.index)
    final_lower = pd.Series(np.nan, index=df.index)
    final_upper.iloc[period-1] = upper_basic.iloc[period-1]
    final_lower.iloc[period-1] = lower_basic.iloc[period-1]
    
    for i in range(period, len(df)):
        fu = upper_basic.iloc[i]
        fl = lower_basic.iloc[i]
        prev_fu = final_upper.iloc[i-1]
        prev_fl = final_lower.iloc[i-1]
        prev_close = close.iloc[i-1]
        
        if np.isnan(prev_fu):
            final_upper.iloc[i] = fu
        elif prev_close > prev_fu:
            final_upper.iloc[i] = min(fu, prev_fu)
        else:
            final_upper.iloc[i] = prev_fu
        
        if np.isnan(prev_fl):
            final_lower.iloc[i] = fl
        elif prev_close < prev_fl:
            final_lower.iloc[i] = max(fl, prev_fl)
        else:
            final_lower.iloc[i] = prev_fl
    
    close_above_lower = (close > final_lower) & (close.shift(1) <= final_lower.shift(1))
    close_below_upper = (close < final_upper) & (close.shift(1) >= final_upper.shift(1))
    
    long_entries = close_above_lower & ~pd.isna(final_lower)
    short_entries = close_below_upper & ~pd.isna(final_upper)
    
    entries = []
    trade_num = 1
    
    for idx in long_entries[long_entries].index:
        ts = int(df['time'].iloc[idx])
        entries.append({
            'trade_num': trade_num,
            'direction': 'long',
            'entry_ts': ts,
            'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            'entry_price_guess': close.iloc[idx],
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': close.iloc[idx],
            'raw_price_b': close.iloc[idx]
        })
        trade_num += 1
    
    for idx in short_entries[short_entries].index:
        ts = int(df['time'].iloc[idx])
        entries.append({
            'trade_num': trade_num,
            'direction': 'short',
            'entry_ts': ts,
            'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            'entry_price_guess': close.iloc[idx],
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': close.iloc[idx],
            'raw_price_b': close.iloc[idx]
        })
        trade_num += 1
    
    return entries