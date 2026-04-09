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
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    true_range.iloc[0] = high.iloc[0] - low.iloc[0]
    
    atr = pd.Series(np.nan, index=df.index)
    atr.iloc[0] = true_range.iloc[0]
    alpha = 1.0 / 144
    for i in range(1, len(df)):
        atr.iloc[i] = true_range.iloc[i] * alpha + atr.iloc[i-1] * (1 - alpha)
    
    fvgTH = 0.5
    atr_filtered = atr * fvgTH
    
    bullG = low > high.shift(1)
    bearG = high < low.shift(1)
    
    bull_fvg = (
        (low - high.shift(2)) > atr_filtered.shift(2).shift(2) & 
        (low > high.shift(2)) & 
        (close.shift(1) > high.shift(2)) & 
        ~bullG & 
        ~bullG.shift(1)
    )
    
    bear_fvg = (
        (low.shift(2) - high) > atr_filtered.shift(2).shift(2) & 
        (high < low.shift(2)) & 
        (close.shift(1) < low.shift(2)) & 
        ~bearG & 
        ~bearG.shift(1)
    )
    
    entries = []
    trade_num = 1
    
    for i in range(2, len(df)):
        if i < 144:
            continue
        if pd.isna(atr.iloc[i]) or pd.isna(atr_filtered.iloc[i-2]):
            continue
            
        if bull_fvg.iloc[i]:
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
        elif bear_fvg.iloc[i]:
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
    
    return entries