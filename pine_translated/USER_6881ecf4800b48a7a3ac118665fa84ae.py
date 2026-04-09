import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    results = []
    trade_num = 1
    
    prd = 2
    df = df.copy()
    df['pivot_high'] = df['high'].rolling(window=prd*2+1, center=True).max()
    df['pivot_low'] = df['low'].rolling(window=prd*2+1, center=True).min()
    df['is_ph'] = (df['high'] == df['pivot_high']) & df['pivot_high'].notna()
    df['is_pl'] = (df['low'] == df['pivot_low']) & df['pivot_low'].notna()
    
    zigzag = []
    last_high = np.nan
    last_low = np.nan
    direction = 0
    
    for i in range(len(df)):
        if df['is_ph'].iloc[i]:
            last_high = df['high'].iloc[i]
            zigzag.insert(0, last_high)
            if len(zigzag) > 50:
                zigzag.pop()
            if direction == -1:
                direction = 1
        if df['is_pl'].iloc[i]:
            last_low = df['low'].iloc[i]
            zigzag.insert(0, last_low)
            if len(zigzag) > 50:
                zigzag.pop()
            if direction == 1:
                direction = -1
        
        fib_50 = np.nan
        if len(zigzag) >= 6:
            fib_0 = zigzag[2]
            fib_1 = zigzag[0]
            diff = fib_1 - fib_0
            fib_50 = fib_0 + diff * 0.5
        
        df.loc[df.index[i], 'fib_50'] = fib_50
    
    df['fib_50'] = df['fib_50'].ffill()
    
    close = df['close']
    fib = df['fib_50']
    
    long_cond = (close > fib) & (close.shift(1) <= fib.shift(1))
    short_cond = (close < fib) & (close.shift(1) >= fib.shift(1))
    
    for i in range(1, len(df)):
        if pd.isna(fib.iloc[i]) or fib.iloc[i] == 0:
            continue
        direction = None
        if long_cond.iloc[i]:
            direction = 'long'
        elif short_cond.iloc[i]:
            direction = 'short'
        
        if direction:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
            results.append({
                'trade_num': trade_num,
                'direction': direction,
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
    
    return results