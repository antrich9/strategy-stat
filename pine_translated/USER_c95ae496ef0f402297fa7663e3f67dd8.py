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
    high = df['high']
    low = df['low']
    close = df['close']
    time = df['time']
    
    # FVG detection
    bull_fvg = (low > high.shift(2)) & (close.shift(1) > high.shift(2))
    bear_fvg = (high < low.shift(2)) & (close.shift(1) < low.shift(2))
    
    # BPR parameters
    lookback_bars = 12
    threshold = 0.0
    
    # Bullish BPR logic
    bull_since = np.zeros(len(df), dtype=int)
    bull_result = pd.Series(False, index=df.index)
    bull_entered_global = False
    
    for i in range(3, len(df)):
        if bull_fvg.iloc[i]:
            bull_since[i] = 0
        elif bull_since[i-1] > 0:
            bull_since[i] = bull_since[i-1] + 1
        else:
            bull_since[i] = 0
        
        bull_cond_1 = bull_fvg.iloc[i] and bull_since[i] <= lookback_bars
        
        if bull_cond_1:
            combined_low_bull = max(high.iloc[i - bull_since[i]], high.iloc[i - 2])
            combined_high_bull = min(low.iloc[i - bull_since[i] + 2], low.iloc[i])
            bull_result.iloc[i] = (combined_high_bull - combined_low_bull >= threshold)
            
            if bull_result.iloc[i] and low.iloc[i] < high.iloc[i - 2]:
                bull_entered_global = True
                if bull_entered_global:
                    ts = int(time.iloc[i])
                    return [{
                        'trade_num': 1,
                        'direction': 'long',
                        'entry_ts': ts,
                        'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                        'entry_price_guess': close.iloc[i],
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': close.iloc[i],
                        'raw_price_b': close.iloc[i]
                    }]
    
    # Bearish BPR logic
    bear_since = np.zeros(len(df), dtype=int)
    bear_result = pd.Series(False, index=df.index)
    bear_entered_global = False
    
    for i in range(3, len(df)):
        if bear_fvg.iloc[i]:
            bear_since[i] = 0
        elif bear_since[i-1] > 0:
            bear_since[i] = bear_since[i-1] + 1
        else:
            bear_since[i] = 0
        
        bear_cond_1 = bear_fvg.iloc[i] and bear_since[i] <= lookback_bars
        
        if bear_cond_1:
            combined_low_bear = max(high.iloc[i - bear_since[i] + 2], high.iloc[i])
            combined_high_bear = min(low.iloc[i - bear_since[i]], low.iloc[i - 2])
            bear_result.iloc[i] = (combined_high_bear - combined_low_bear >= threshold)
            
            if bear_result.iloc[i] and high.iloc[i] > low.iloc[i - 2]:
                bear_entered_global = True
                if bear_entered_global:
                    ts = int(time.iloc[i])
                    return [{
                        'trade_num': 1,
                        'direction': 'short',
                        'entry_ts': ts,
                        'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                        'entry_price_guess': close.iloc[i],
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': close.iloc[i],
                        'raw_price_b': close.iloc[i]
                    }]
    
    return []