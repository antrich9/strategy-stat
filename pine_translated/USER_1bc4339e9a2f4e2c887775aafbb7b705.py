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
    
    high = df['high']
    low = df['low']
    close = df['close']
    time_col = df['time']
    
    # Calculate Wilder ATR(144)
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/144, adjust=False).mean()
    
    # Bullish/Bearish gaps
    bullG = low > high.shift(1)
    bearG = high < low.shift(1)
    
    # Bull condition
    bull = (
        (low - high.shift(2)) > atr &
        (low > high.shift(2)) &
        (close.shift(1) > high.shift(2)) &
        ~(bullG | bullG.shift(1))
    )
    
    # Bear condition
    bear = (
        (low.shift(2) - high) > atr &
        (high < low.shift(2)) &
        (close.shift(1) < low.shift(2)) &
        ~(bearG | bearG.shift(1))
    )
    
    # Time window filtering
    dt = pd.to_datetime(time_col, unit='s', utc=True).dt.tz_convert('Europe/London')
    total_minutes = dt.dt.hour * 60 + dt.dt.minute
    
    morning_window = (total_minutes >= 6 * 60 + 45) & (total_minutes < 9 * 60 + 45)
    afternoon_window = (total_minutes >= 14 * 60 + 45) & (total_minutes < 16 * 60 + 45)
    in_trading_window = morning_window | afternoon_window
    
    # Apply time window filter
    valid_long = bull & in_trading_window & ~atr.isna()
    valid_short = bear & in_trading_window & ~atr.isna()
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if valid_long.iloc[i]:
            entry_ts = int(time_col.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
    
    for i in range(len(df)):
        if valid_short.iloc[i]:
            entry_ts = int(time_col.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
    
    return entries