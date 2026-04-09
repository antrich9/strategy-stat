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
    
    london_morning_start = datetime(2020, 1, 1, 8, 0, tzinfo=timezone.utc)
    london_morning_end = datetime(2020, 1, 1, 9, 55, tzinfo=timezone.utc)
    london_afternoon_start = datetime(2020, 1, 1, 14, 0, tzinfo=timezone.utc)
    london_afternoon_end = datetime(2020, 1, 1, 16, 55, tzinfo=timezone.utc)
    
    times = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_convert('Europe/London')
    hours = times.dt.hour
    minutes = times.dt.minute
    total_minutes = hours * 60 + minutes
    
    morning_start_min = 8 * 60
    morning_end_min = 9 * 60 + 55
    afternoon_start_min = 14 * 60
    afternoon_end_min = 16 * 60 + 55
    
    is_within_morning = (total_minutes >= morning_start_min) & (total_minutes <= morning_end_min)
    is_within_afternoon = (total_minutes >= afternoon_start_min) & (total_minutes <= afternoon_end_min)
    in_trading_window = is_within_morning | is_within_afternoon
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    swing_high = (high > high.shift(1)) & (high > high.shift(-1))
    swing_low = (low < low.shift(1)) & (low < low.shift(-1))
    
    atr_length = 14
    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1.0/atr_length, adjust=False).mean()
    
    bullish_fvg = low > high.shift(1)
    bearish_fvg = high < low.shift(1)
    
    long_condition = bullish_fvg & swing_low.shift(1) & in_trading_window & ~atr.isna()
    short_condition = bearish_fvg & swing_high.shift(1) & in_trading_window & ~atr.isna()
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if long_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
        
        if short_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
    
    return entries