import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    fastLength = 50
    slowLength = 200
    
    fastEMA = df['close'].ewm(span=fastLength, adjust=False).mean()
    slowEMA = df['close'].ewm(span=slowLength, adjust=False).mean()
    
    crossover = (fastEMA > slowEMA) & (fastEMA.shift(1) <= slowEMA.shift(1))
    crossunder = (fastEMA < slowEMA) & (fastEMA.shift(1) >= slowEMA.shift(1))
    
    entries = []
    
    for i in range(1, len(df)):
        timestamp = df['time'].iloc[i]
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        
        is_long_time = (7 <= dt.hour < 10) or (12 <= dt.hour < 15)
        is_short_time = is_long_time
        
        if crossover.iloc[i] and is_long_time:
            entries.append({
                'trade_num': len(entries) + 1,
                'direction': 'long',
                'entry_ts': timestamp,
                'entry_time': dt.isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
        
        if crossunder.iloc[i] and is_short_time:
            entries.append({
                'trade_num': len(entries) + 1,
                'direction': 'short',
                'entry_ts': timestamp,
                'entry_time': dt.isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
    
    return entries