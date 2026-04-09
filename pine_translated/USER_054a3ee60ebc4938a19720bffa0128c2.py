import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['ts'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Resample to 1H for FVG conditions (mimics request.security with "60")
    h1 = df.set_index('ts').resample('1H').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna().copy()
    
    # Resample to 1D for swing detection
    d1 = df.set_index('ts').resample('1D').agg({
        'high': 'max', 'low': 'min'
    }).dropna().copy()
    
    # FVG conditions using Pine Script offset logic (high[1], high[2], low[2] -> shift(1), shift(2), shift(2))
    h1['bfvg_condition'] = (h1['high'].shift(1) > h1['low'].shift(3))
    h1['sfvg_condition'] = (h1['high'].shift(1) < h1['low'].shift(3))
    
    # Swing detection on daily data
    d1['is_swing_high'] = (d1['high'].shift(1) < d1['high'].shift(2)) & \
                          (d1['high'].shift(0) < d1['high'].shift(2)) & \
                          (d1['high'].shift(-3) < d1['high'].shift(2)) & \
                          (d1['high'].shift(-4) < d1['high'].shift(2))
    d1['is_swing_low'] = (d1['low'].shift(1) > d1['low'].shift(2)) & \
                         (d1['low'].shift(0) > d1['low'].shift(2)) & \
                         (d1['low'].shift(-3) > d1['low'].shift(2)) & \
                         (d1['low'].shift(-4) > d1['low'].shift(2))
    
    d1['lastSwingType'] = 'none'
    d1.loc[d1['is_swing_high'], 'lastSwingType'] = 'dailyHigh'
    d1.loc[d1['is_swing_low'], 'lastSwingType'] = 'dailyLow'
    d1['lastSwingType'] = d1['lastSwingType'].replace('none', np.nan).ffill().fillna('none')
    
    # Merge 1H with daily swing type
    h1['date'] = h1.index.date
    d1['date'] = d1.index.date
    h1 = h1.reset_index().merge(d1[['date', 'lastSwingType']], on='date', how='left')
    h1['lastSwingType'] = h1['lastSwingType'].ffill().fillna('none')
    h1['bfvg_confirmed'] = h1['bfvg_condition'] & (h1['lastSwingType'] == 'dailyLow')
    h1['sfvg_confirmed'] = h1['sfvg_condition'] & (h1['lastSwingType'] == 'dailyHigh')
    
    # Trading window (London: 06:45-08:45 and 13:00-13:45 UTC, with BST adjustment)
    def in_trading_window(ts):
        h = ts.hour
        m = ts.minute
        t = h * 60 + m
        return (405 <= t <= 525) or (780 <= t <= 825) or \
               (465 <= t <= 585) or (840 <= t <= 885)
    
    h1['in_window'] = h1['ts'].apply(in_trading_window)
    
    # Bullish: bfvg_confirmed AND in_window
    h1['bull_entry'] = h1['bfvg_confirmed'] & h1['in_window']
    # Bearish: sfvg_confirmed AND in_window
    h1['bear_entry'] = h1['sfvg_confirmed'] & h1['in_window']
    
    # Map 1H conditions to 5min
    h1_ts_set = set(h1.loc[h1['bull_entry'], 'ts'])
    h1_bear_ts_set = set(h1.loc[h1['bear_entry'], 'ts'])
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        row = df.iloc[i]
        ts = row['ts']
        entry_price = row['close']
        
        is_bull = ts in h1_ts_set
        is_bear = ts in h1_bear_ts_set
        
        if is_bull or is_bear:
            direction = 'long' if is_bull else 'short'
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': int(row['time']),
                'entry_time': ts.replace(tzinfo=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
    
    return entries