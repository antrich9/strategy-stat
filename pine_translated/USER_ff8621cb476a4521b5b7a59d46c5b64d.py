import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    df columns: time(int unix ts), open(float), high(float), low(float), close(float), volume(float)
    Rows sorted ascending by time (oldest first). Index is 0-based int.

    Returns list of dicts:
    [{'trade_num': int, 'direction': 'long' or 'short',
      'entry_ts': int, 'entry_time': str,
      'entry_price_guess': float,
      'exit_ts': 0, 'exit_time': '', 'exit_price_guess': 0.0,
      'raw_price_a': float, 'raw_price_b': float}]
    """
    
    # Convert time to datetime for grouping by day
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['date'] = df['datetime'].dt.date
    
    # Calculate current timeframe EMAs (cema9, cema18)
    df['cema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['cema18'] = df['close'].ewm(span=18, adjust=False).mean()
    
    # Calculate daily timeframe EMAs (ema9, ema18)
    # Group by date and take the last close of each day for daily EMA calculation
    daily_df = df.groupby('date').agg({
        'time': 'last',
        'close': 'last'
    }).reset_index(drop=True)
    
    # Calculate EMAs on daily data
    ema9_daily = daily_df['close'].ewm(span=9, adjust=False).mean()
    ema18_daily = daily_df['close'].ewm(span=18, adjust=False).mean()
    
    # Map daily EMAs back to intraday dataframe
    daily_df['ema9_daily'] = ema9_daily
    daily_df['ema18_daily'] = ema18_daily
    
    # Merge daily EMAs to intraday df using date as key
    daily_emas = daily_df[['date', 'ema9_daily', 'ema18_daily']].copy()
    df = df.merge(daily_emas, on='date', how='left', suffixes=('', '_dup'))
    
    # Remove duplicate column if exists
    if 'ema9_daily_dup' in df.columns:
        df.drop(['ema9_daily_dup', 'ema18_daily_dup'], axis=1, inplace=True)
    
    # Forward fill daily EMAs to intraday dataframe
    df['ema9_daily'] = df['ema9_daily'].ffill()
    df['ema18_daily'] = df['ema18_daily'].ffill()
    
    # Fill NaN with current close for bars before first daily close
    df['ema9_daily'] = df['ema9_daily'].fillna(df['close'])
    df['ema18_daily'] = df['ema18_daily'].fillna(df['close'])
    
    # Compute entry conditions
    condition_long = (df['close'] > df['ema9_daily']) & (df['close'] > df['ema18_daily']) & (df['close'] > df['cema9']) & (df['close'] > df['cema18'])
    condition_short = (df['close'] < df['ema9_daily']) & (df['close'] < df['ema18_daily']) & (df['close'] < df['cema9']) & (df['close'] < df['cema18'])
    
    # Build boolean Series for crossover conditions
    long_cond = condition_long & ~condition_long.shift(1).fillna(False)
    short_cond = condition_short & ~condition_short.shift(1).fillna(False)
    
    # Iterate through bars to generate entries
    entries = []
    trade_num = 1
    
    for i in range(1, len(df)):
        # Skip if required indicators are NaN
        if pd.isna(df['cema9'].iloc[i]) or pd.isna(df['cema18'].iloc[i]) or pd.isna(df['ema9_daily'].iloc[i]) or pd.isna(df['ema18_daily'].iloc[i]):
            continue
        
        if long_cond.iloc[i]:
            ts = df['time'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif short_cond.iloc[i]:
            ts = df['time'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(ts),
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