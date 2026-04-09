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
    entries = []
    trade_num = 0
    
    close = df['close']
    open_vals = df['open']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Helper functions for OB/FVG detection
    def is_up(idx):
        return close.iloc[idx] > open_vals.iloc[idx]
    
    def is_down(idx):
        return close.iloc[idx] < open_vals.iloc[idx]
    
    def is_ob_up(idx):
        return is_down(idx + 1) and is_up(idx) and close.iloc[idx] > high.iloc[idx + 1]
    
    def is_ob_down(idx):
        return is_up(idx + 1) and is_down(idx) and close.iloc[idx] < low.iloc[idx + 1]
    
    def is_fvg_up(idx):
        return low.iloc[idx] > high.iloc[idx + 2]
    
    def is_fvg_down(idx):
        return high.iloc[idx] < low.iloc[idx + 2]
    
    # Filters
    vol_filt = volume.shift(1) > volume.rolling(9).mean() * 1.5
    
    atr_20 = df['high'].rolling(20).max() - df['low'].rolling(20).min()
    atr_filt = (low - high.shift(2) > atr_20) | (low.shift(2) - high > atr_20)
    
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    
    # Build boolean series for entry conditions
    ob_up_series = pd.Series([False] * len(df), index=df.index)
    ob_down_series = pd.Series([False] * len(df), index=df.index)
    fvg_up_series = pd.Series([False] * len(df), index=df.index)
    fvg_down_series = pd.Series([False] * len(df), index=df.index)
    
    # Detect OB and FVG conditions
    for i in range(3, len(df)):
        try:
            ob_up_series.iloc[i] = is_ob_up(i - 1)
            ob_down_series.iloc[i] = is_ob_down(i - 1)
        except:
            pass
        try:
            fvg_up_series.iloc[i] = is_fvg_up(i - 1)
            fvg_down_series.iloc[i] = is_fvg_down(i - 1)
        except:
            pass
    
    # Additional conditions
    top_imbalance_bway = (low.shift(2) <= open_vals.shift(1)) & (high >= close.shift(1)) & (close < low.shift(1))
    bottom_inbalance_bway = (high.shift(2) >= open_vals.shift(1)) & (low <= close.shift(1)) & (close > high.shift(1))
    
    # Breakaway FVG conditions
    bfvg = (low > high.shift(2)) & vol_filt & atr_filt & loc2
    sfvg = (high < low.shift(2)) & vol_filt & atr_filt & ~loc2
    
    # Long entry condition: OB+FVG stacked bullish or breakaway FVG with conditions
    long_condition = (ob_up_series & fvg_up_series) | (bfvg & top_imbalance_bway)
    
    # Short entry condition: OB+FVG stacked bearish or breakaway FVG with conditions
    short_condition = (ob_down_series & fvg_down_series) | (sfvg & bottom_inbalance_bway)
    
    # Iterate through bars and generate entries
    for i in range(1, len(df)):
        if pd.isna(long_condition.iloc[i]) or pd.isna(short_condition.iloc[i]):
            continue
        
        ts = int(df['time'].iloc[i])
        entry_price = close.iloc[i]
        
        if long_condition.iloc[i]:
            trade_num += 1
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
        
        if short_condition.iloc[i]:
            trade_num += 1
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
    
    return entries