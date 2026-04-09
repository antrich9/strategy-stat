import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    lookback = 14
    
    # Calculate reference high (highest high over lookback period)
    ref_high = df['high'].rolling(window=lookback, min_periods=lookback).max()
    
    # Calculate dip price (80% of reference high)
    dip_price = ref_high * 0.80
    
    entries = []
    trade_num = 1
    in_position = False
    
    for i in range(len(df)):
        # Skip bars where indicators are NaN
        if pd.isna(ref_high.iloc[i]) or pd.isna(dip_price.iloc[i]):
            continue
        
        # Entry condition: not in position AND close < dip_price
        if not in_position and df['close'].iloc[i] < dip_price.iloc[i]:
            ts = df['time'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(dip_price.iloc[i]),
                'raw_price_b': float(dip_price.iloc[i])
            })
            trade_num += 1
            in_position = True
    
    return entries