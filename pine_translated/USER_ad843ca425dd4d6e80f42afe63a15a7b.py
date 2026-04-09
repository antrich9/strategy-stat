import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Ensure sorted by time
    df = df.sort_values('time').reset_index(drop=True)
    
    # Compute EMAs
    ema9 = df['close'].ewm(span=9, adjust=False).mean()
    ema18 = df['close'].ewm(span=18, adjust=False).mean()
    
    # Entry conditions
    condition_long = (df['close'] > ema9) & (df['close'] > ema18)
    condition_short = (df['close'] < ema9) & (df['close'] < ema18)
    
    # Detect start of condition (crossover)
    # We use shift(1) to get previous bar condition
    prev_condition_long = condition_long.shift(1).fillna(False)
    prev_condition_short = condition_short.shift(1).fillna(False)
    
    long_entry = condition_long & ~prev_condition_long
    short_entry = condition_short & ~prev_condition_short
    
    # Optionally, we could also consider other conditions like obUp, fvgUp, but we stick to EMA condition.
    
    entries = []
    trade_num = 1
    
    for i in range(1, len(df)):
        # Skip if any required indicator is NaN
        if pd.isna(ema9.iloc[i]) or pd.isna(ema18.iloc[i]):
            continue
        
        if long_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry = {
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            }
            entries.append(entry)
            trade_num += 1
        elif short_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry = {
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            }
            entries.append(entry)
            trade_num += 1
    
    return entries