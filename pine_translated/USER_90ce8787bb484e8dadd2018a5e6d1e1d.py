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
    
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('datetime', inplace=True)
    
    # Get daily close for EMA calculation (request.security equivalent)
    daily_close = df['close'].resample('D').last().dropna()
    
    # Calculate daily EMAs: ta.ema(src, len) = src.ewm(span=len, adjust=False).mean()
    ema9 = daily_close.ewm(span=9, adjust=False).mean()
    ema18 = daily_close.ewm(span=18, adjust=False).mean()
    
    # Merge daily EMAs back to original timeframe using forward fill
    df['ema9'] = ema9.reindex(df.index, method='ffill')
    df['ema18'] = ema18.reindex(df.index, method='ffill')
    
    # Define entry conditions from Pine Script:
    # condition_long = (close > ema9) and (close > ema18)
    # condition_short = (close < ema9) and (close < ema18)
    condition_long = (df['close'] > df['ema9']) & (df['close'] > df['ema18'])
    condition_short = (df['close'] < df['ema9']) & (df['close'] < df['ema18'])
    
    # Build boolean Series for crossover detection (ta.crossover)
    # prev_close > prev_ema9 & prev_ema18 is the negation of condition_long in previous bar
    prev_close = df['close'].shift(1)
    prev_ema9 = df['ema9'].shift(1)
    prev_ema18 = df['ema18'].shift(1)
    prev_condition_long = (prev_close > prev_ema9) & (prev_close > prev_ema18)
    prev_condition_short = (prev_close < prev_ema9) & (prev_close < prev_ema18)
    
    # ta.crossover(condition_long, condition_long[1]) at bar i
    long_crossover = condition_long & ~prev_condition_long
    short_crossover = condition_short & ~prev_condition_short
    
    # Skip bars where required indicators are NaN
    valid = df['ema9'].notna() & df['ema18'].notna() & df['close'].notna()
    
    entries = []
    trade_num = 1
    
    # Iterate with for loop (rule 14)
    for i in range(len(df)):
        if not valid.iloc[i]:
            continue
        
        if long_crossover.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif short_crossover.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries