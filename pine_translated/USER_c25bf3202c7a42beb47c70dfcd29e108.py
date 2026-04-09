import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Compute indicators
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - low.shift(1))
    tr3 = np.abs(low - high.shift(1))
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder ATR (144)
    atr = true_range.ewm(alpha=1/144, adjust=False).mean()
    
    # ATR threshold (fvgTH = 0.5)
    atr_threshold = atr * 0.5
    
    # Bullish and Bearish gaps
    bullG = low > high.shift(1)
    bearG = high < low.shift(1)
    
    # Shifted versions
    bullG_shift1 = bullG.shift(1)
    bearG_shift1 = bearG.shift(1)
    
    # Bull condition
    bull = (low - high.shift(2)) > atr_threshold & (low > high.shift(2)) & (close.shift(1) > high.shift(2)) & ~(bullG | bullG_shift1)
    
    # Actually, we need parentheses correctly. Use:
    bull = ((low - high.shift(2)) > atr_threshold) & (low > high.shift(2)) & (close.shift(1) > high.shift(2)) & ~(bullG | bullG_shift1)
    
    # Bear condition
    bear = ((low.shift(2) - high) > atr_threshold) & (high < low.shift(2)) & (close.shift(1) < low.shift(2)) & ~(bearG | bearG_shift1)
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if bull.iloc[i] or bear.iloc[i]:
            direction = 'long' if bull.iloc[i] else 'short'
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = close.iloc[i]
            
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries