import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['ts'] = df['time']
    df['date'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    df['prev_day'] = df['date'].dt.date
    
    daily_hlc = df.groupby('prev_day').agg({
        'high': 'max',
        'low': 'min'
    }).reset_index()
    daily_hlc.columns = ['prev_day', 'prev_day_high', 'prev_day_low']
    
    df = df.merge(daily_hlc, on='prev_day', how='left')
    df['prev_day_high'] = df['prev_day_high'].ffill()
    df['prev_day_low'] = df['prev_day_low'].ffill()
    
    prev_day_high = df['prev_day_high'].shift(1)
    prev_day_low = df['prev_day_low'].shift(1)
    
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    
    minute_of_day = df['hour'] * 60 + df['minute']
    in_trading_window = ((minute_of_day >= 420) & (minute_of_day <= 659)) | ((minute_of_day >= 900) & (minute_of_day <= 1019))
    
    flagpdh = (df['close'] > prev_day_high) & ~prev_day_high.isna()
    flagpdl = (df['close'] < prev_day_low) & ~prev_day_low.isna()
    
    entries = []
    trade_num = 1
    
    for i in range(1, len(df)):
        if in_trading_window.iloc[i] and flagpdh.iloc[i]:
            entry_ts = int(df['ts'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        
        if in_trading_window.iloc[i] and flagpdl.iloc[i]:
            entry_ts = int(df['ts'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries