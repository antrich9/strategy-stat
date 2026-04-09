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
    df = df.copy()
    df['time_dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = df['time_dt'].dt.hour
    
    isValidTradeTime = (df['hour'] >= 10) & (df['hour'] < 12)
    
    bullishFVG = (df['low'] > df['high'].shift(2)) & (df['close'].shift(1) > df['high'].shift(2))
    bearishFVG = (df['high'] < df['low'].shift(2)) & (df['close'].shift(1) < df['low'].shift(2))
    
    lookback = 30
    entries = []
    trade_num = 1
    
    for i in range(2, len(df)):
        if not isValidTradeTime.iloc[i]:
            continue
        if pd.isna(bullishFVG.iloc[i]) or pd.isna(bearishFVG.iloc[i]):
            continue
        
        if bullishFVG.iloc[i]:
            bull_since_vals = []
            for j in range(1, min(lookback, i)):
                if bearishFVG.iloc[i-j]:
                    bull_since_vals.append(j)
            if bull_since_vals:
                bear_since = bull_since_vals[0]
                if bear_since <= lookback:
                    cond_valid = True
                    if cond_valid:
                        bar_index = i
                        bar_index_2 = i - 2
                        p1 = df['high'].iloc[i-2]
                        p2 = df['low'].iloc[i]
                        found_bpr = False
                        for k in range(2, min(lookback, i)):
                            idx = i - k
                            if idx >= 0 and bearishFVG.iloc[idx]:
                                if p1 < df['low'].iloc[idx+2] if idx + 2 < len(df) else False:
                                    if p2 > df['high'].iloc[idx]:
                                        found_bpr = True
                                        break
                        if found_bpr:
                            ts = int(df['time'].iloc[i])
                            entries.append({
                                'trade_num': trade_num,
                                'direction': 'long',
                                'entry_ts': ts,
                                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                                'entry_price_guess': float(df['close'].iloc[i]),
                                'exit_ts': 0,
                                'exit_time': '',
                                'exit_price_guess': 0.0,
                                'raw_price_a': float(df['close'].iloc[i]),
                                'raw_price_b': float(df['close'].iloc[i])
                            })
                            trade_num += 1
    
    return entries