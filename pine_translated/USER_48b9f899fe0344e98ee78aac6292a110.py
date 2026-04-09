import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    
    # Convert time to datetime for processing
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['date'] = df['datetime'].dt.date
    
    # Calculate daily high and low to get previous day values
    daily_agg = df.groupby('date').agg(
        daily_high=('high', 'max'),
        daily_low=('low', 'min')
    ).reset_index()
    
    # Shift to get previous day's high and low
    daily_agg['prev_day_high'] = daily_agg['daily_high'].shift(1)
    daily_agg['prev_day_low'] = daily_agg['daily_low'].shift(1)
    
    # Merge previous day values back to main dataframe
    df = df.merge(daily_agg[['date', 'prev_day_high', 'prev_day_low']], on='date', how='left')
    
    # Detect Previous Day High raid (hit) - signal for short entry
    # Short entry when price hits or exceeds previous day high
    df['pdh_hit'] = (df['high'] >= df['prev_day_high']) & df['prev_day_high'].notna()
    
    # Only trigger on first hit per day to avoid duplicate entries
    df['prev_pdh_hit'] = df.groupby('date')['pdh_hit'].shift(1).fillna(False)
    df['new_pdh_hit'] = df['pdh_hit'] & ~df['prev_pdh_hit']
    
    # Build entries list
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if df['new_pdh_hit'].iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries