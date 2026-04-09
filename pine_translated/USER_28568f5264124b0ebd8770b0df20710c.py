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
    # Calculate daily EMAs (9 and 18) from daily close prices
    df_copy = df.copy()
    df_copy['datetime'] = pd.to_datetime(df_copy['time'], unit='s', utc=True)
    df_copy['date'] = df_copy['datetime'].dt.date
    
    # Get daily close prices
    daily_df = df_copy.groupby('date')['close'].last()
    
    # Calculate EMAs on daily data
    ema9_daily = daily_df.ewm(span=9, adjust=False).mean()
    ema18_daily = daily_df.ewm(span=18, adjust=False).mean()
    
    # Create mappings from date to EMA values
    ema9_by_date = ema9_daily.to_dict()
    ema18_by_date = ema18_daily.to_dict()
    
    # Expand daily EMA values to hourly bars
    ema9 = df['time'].map(lambda x: ema9_by_date.get(pd.to_datetime(x, unit='s', utc=True).date(), np.nan))
    ema18 = df['time'].map(lambda x: ema18_by_date.get(pd.to_datetime(x, unit='s', utc=True).date(), np.nan))
    
    # Time window conditions (London time)
    # Morning: 07:45 to 09:45
    # Afternoon: 14:45 to 16:45
    morning_start = 7 * 60 + 45
    morning_end = 9 * 60 + 45
    afternoon_start = 14 * 60 + 45
    afternoon_end = 16 * 60 + 45
    
    hours = df['time'].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc).hour)
    minutes = df['time'].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc).minute)
    time_in_minutes = hours * 60 + minutes
    
    in_morning_window = (time_in_minutes >= morning_start) & (time_in_minutes <= morning_end)
    in_afternoon_window = (time_in_minutes >= afternoon_start) & (time_in_minutes <= afternoon_end)
    in_window = in_morning_window | in_afternoon_window
    
    # FVG conditions
    bfvg = df['low'] > df['high'].shift(2)
    sfvg = df['high'] < df['low'].shift(2)
    
    # Entry conditions
    condition_long = (df['close'] > ema9) & (df['close'] > ema18)
    condition_short = (df['close'] < ema9) & (df['close'] < ema18)
    
    long_condition = in_window & bfvg & condition_long
    short_condition = in_window & sfvg & condition_short
    
    # Generate entries
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
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif short_condition.iloc[i]:
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