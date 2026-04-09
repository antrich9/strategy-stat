import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    high = df['high']
    low = df['low']
    close = df['close']
    
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1.0/144, adjust=False).mean()
    
    fvgTH = 0.5
    atr_filter = atr * fvgTH
    
    bullG = low > high.shift(1)
    bearG = high < low.shift(1)
    
    bull = (low - high.shift(2)) > atr_filter
    bull = bull & (low > high.shift(2))
    bull = bull & (close.shift(1) > high.shift(2))
    bull = bull & ~(bullG | bullG.shift(1))
    
    bear = (low.shift(2) - high) > atr_filter
    bear = bear & (high < low.shift(2))
    bear = bear & (close.shift(1) < low.shift(2))
    bear = bear & ~(bearG | bearG.shift(1))
    
    results = []
    trade_num = 1
    
    for i in range(2, len(df)):
        if pd.isna(atr.iloc[i]) or pd.isna(atr_filter.iloc[i]):
            continue
            
        if bull.iloc[i]:
            ts = int(df['time'].iloc[i])
            results.append({
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
        elif bear.iloc[i]:
            ts = int(df['time'].iloc[i])
            results.append({
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
    
    return results