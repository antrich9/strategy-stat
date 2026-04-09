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
    # Series shortcuts
    open_s  = df['open']
    high_s  = df['high']
    low_s   = df['low']
    close_s = df['close']

    # Previous bar values
    prev_open  = open_s.shift(1)
    prev_high  = high_s.shift(1)
    prev_low   = low_s.shift(1)
    prev_close = close_s.shift(1)

    # Bullish 2CR components
    bull_reject = (prev_close > prev_open) & ((prev_open - prev_low) > 2 * np.abs(prev_close - prev_open))
    bull_confirm = (low_s < prev_low) & (close_s > prev_close)
    bull_2cr = bull_reject & bull_confirm

    # Bearish 2CR components
    bear_reject = (prev_close < prev_open) & ((prev_high - prev_open) > 2 * np.abs(prev_close - prev_open))
    bear_confirm = (high_s > prev_high) & (close_s < prev_close)
    bear_2cr = bear_reject & bear_confirm

    entries = []
    trade_num = 1

    # Start from bar 1 because we need previous bar data
    for i in range(1, len(df)):
        if bull_2cr.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif bear_2cr.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries