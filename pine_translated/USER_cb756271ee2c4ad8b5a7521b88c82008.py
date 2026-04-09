import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    df['date'] = df['datetime'].dt.date
    
    daily_agg = df.groupby('date').agg(
        daily_open=('open', 'first'),
        daily_high=('high', 'max'),
        daily_low=('low', 'min'),
        daily_close=('close', 'last')
    ).reset_index()
    
    df = df.merge(daily_agg, on='date')
    
    df['daily_dir'] = np.where(df['daily_close'] > df['daily_open'], 1,
                     np.where(df['daily_close'] < df['daily_open'], -1, 0))
    df['daily_dir'] = df['daily_dir'].fillna(0)
    
    df['prev_daily_dir'] = df['daily_dir'].shift(1).fillna(0)
    
    bullish_crossover = (df['daily_dir'] == 1) & (df['prev_daily_dir'] != 1)
    bearish_crossover = (df['daily_dir'] == -1) & (df['prev_daily_dir'] != -1)
    
    long_condition = bullish_crossover & (df['low'] < df['daily_low'])
    short_condition = bearish_crossover & (df['high'] > df['daily_high'])
    
    if df['low'].isna().any():
        long_condition = long_condition.fillna(False)
    if df['high'].isna().any():
        short_condition = short_condition.fillna(False)
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if long_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
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
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
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
            })
            trade_num += 1
    
    return entries