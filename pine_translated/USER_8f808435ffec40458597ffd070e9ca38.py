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
    left = 20
    right = 15
    nPiv = 4
    atrLen = 30
    mult = 0.5
    per = 5.0
    
    if len(df) < max(left + right + 1, atrLen + 1):
        return []
    
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    time = df['time'].values
    
    # Wilder ATR
    tr1 = np.abs(high - low)
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[0], np.maximum(np.maximum(tr1[1:], tr2), tr3)])
    atr = np.zeros(len(df))
    atr[atrLen - 1] = np.mean(tr[:atrLen])
    for i in range(atrLen, len(df)):
        atr[i] = (atr[i - 1] * (atrLen - 1) + tr[i]) / atrLen
    
    # Find pivots
    pivots_high_ts = []
    pivots_high_price = []
    pivots_low_ts = []
    pivots_low_price = []
    
    for i in range(left, len(df) - right):
        is_high = True
        for j in range(i - left, i + right + 1):
            if j != i and high[i] <= high[j]:
                is_high = False
                break
        if is_high:
            pivots_high_ts.append(time[i])
            pivots_high_price.append(high[i])
        
        is_low = True
        for j in range(i - left, i + right + 1):
            if j != i and low[i] >= low[j]:
                is_low = False
                break
        if is_low:
            pivots_low_ts.append(time[i])
            pivots_low_price.append(low[i])
    
    entries = []
    trade_num = 1
    
    max_per = per / 100.0
    
    for i in range(atrLen, len(df)):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
        
        # Get recent pivots (last nPiv of each type)
        recent_highs = [(pivots_high_ts[j], pivots_high_price[j]) for j in range(len(pivots_high_ts)) if pivots_high_ts[j] < time[i]]
        recent_lows = [(pivots_low_ts[j], pivots_low_price[j]) for j in range(len(pivots_low_ts)) if pivots_low_ts[j] < time[i]]
        
        if len(recent_highs) > nPiv:
            recent_highs = recent_highs[-nPiv:]
        if len(recent_lows) > nPiv:
            recent_lows = recent_lows[-nPiv:]
        
        if len(recent_highs) == 0 or len(recent_lows) == 0:
            continue
        
        # Check max zone size
        if close[i] > 0:
            zone_size = atr[i] * mult * 2
            if zone_size / close[i] > max_per:
                continue
        
        # Breakout above resistance
        highest_high = max([h[1] for h in recent_highs])
        breakout_long = close[i] > highest_high
        
        # Breakdown below support
        lowest_low = min([l[1] for l in recent_lows])
        breakout_short = close[i] < lowest_low
        
        if breakout_long:
            entry_ts = int(time[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(close[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close[i]),
                'raw_price_b': float(close[i])
            })
            trade_num += 1
        elif breakout_short:
            entry_ts = int(time[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(close[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close[i]),
                'raw_price_b': float(close[i])
            })
            trade_num += 1
    
    return entries