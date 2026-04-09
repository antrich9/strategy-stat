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
    # Copy to avoid mutating the input DataFrame
    df = df.copy()

    # Ensure time column is integer timestamps
    df['time'] = df['time'].astype(int)

    # Shifted series for FVG detection
    df['high_2'] = df['high'].shift(2)
    df['low_2'] = df['low'].shift(2)
    df['close_1'] = df['close'].shift(1)

    # Bullish and Bearish Fair Value Gap (FVG) conditions
    df['bull_fvg'] = (df['low'] > df['high_2']) & (df['close_1'] > df['high_2'])
    df['bear_fvg'] = (df['high'] < df['low_2']) & (df['close_1'] < df['low_2'])

    # Convert timestamps to London time for the trading window
    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert('Europe/London')
    df['hour'] = df['dt'].dt.hour
    df['minute'] = df['dt'].dt.minute

    # London morning (08:00‑09:55) and afternoon (14:00‑16:55) windows
    morning = (df['hour'] == 8) | ((df['hour'] == 9) & (df['minute'] <= 55))
    afternoon = (df['hour'] >= 14) & ((df['hour'] < 16) | ((df['hour'] == 16) & (df['minute'] <= 55)))
    df['in_time_window'] = morning | afternoon

    # Entry signals
    long_signal = df['bull_fvg'] & df['in_time_window']
    short_signal = df['bear_fvg'] & df['in_time_window']

    # Build the list of entries
    entries = []
    trade_num = 1
    for i, row in df.iterrows():
        if pd.isna(row['bull_fvg']):
            continue
        if long_signal.loc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(row['time']),
                'entry_time': datetime.fromtimestamp(row['time'], tz=timezone.utc).isoformat(),
                'entry_price_guess': row['close'],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': row['close'],
                'raw_price_b': row['close']
            })
            trade_num += 1
        elif short_signal.loc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(row['time']),
                'entry_time': datetime.fromtimestamp(row['time'], tz=timezone.utc).isoformat(),
                'entry_price_guess': row['close'],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': row['close'],
                'raw_price_b': row['close']
            })
            trade_num += 1

    return entries