import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    
    entries = []
    trade_num = 1
    
    for i in range(1, len(df)):
        if pd.isna(ema8.iloc[i]) or pd.isna(ema20.iloc[i]) or pd.isna(ema50.iloc[i]):
            continue
        
        is_long = (ema8.iloc[i] > ema20.iloc[i] and ema8.iloc[i-1] <= ema20.iloc[i-1]) and ema20.iloc[i] > ema50.iloc[i]
        is_short = (ema8.iloc[i] < ema20.iloc[i] and ema8.iloc[i-1] >= ema20.iloc[i-1]) and ema20.iloc[i] < ema50.iloc[i]
        
        if is_long:
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
        elif is_short:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
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