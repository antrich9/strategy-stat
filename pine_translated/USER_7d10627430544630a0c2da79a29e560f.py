import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    
    close = df['close']
    
    # Current timeframe EMAs
    cema9 = close.ewm(span=9, adjust=False).mean()
    cema18 = close.ewm(span=18, adjust=False).mean()
    
    # 240 (4h) timeframe EMAs - resample to 4h then reindex
    resampled_240 = df.resample('240T').agg({'close': 'last'})
    ema9_240 = resampled_240['close'].ewm(span=9, adjust=False).mean()
    ema18_240 = resampled_240['close'].ewm(span=18, adjust=False).mean()
    
    ema9 = ema9_240.reindex(df.index, method='ffill')
    ema18 = ema18_240.reindex(df.index, method='ffill')
    
    # Long: close above all 4 EMAs
    condition_long = (close > ema9) & (close > ema18) & (close > cema9) & (close > cema18)
    # Short: close below all 4 EMAs
    condition_short = (close < ema9) & (close < ema18) & (close < cema9) & (close < cema18)
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if i < 18:
            continue
        
        if condition_long.iloc[i]:
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
        elif condition_short.iloc[i]:
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