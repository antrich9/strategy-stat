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
    # Ensure dataframe is sorted by time
    df = df.sort_values('time').reset_index(drop=True)

    # Convert timestamps to datetime (UTC) and then to EST (GMT-5)
    df['datetime_utc'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['datetime_est'] = df['datetime_utc'].dt.tz_convert('Etc/GMT+5')
    df['date_est'] = df['datetime_est'].dt.date

    # Compute session open price for each day (first open of the day)
    session_open = df.groupby('date_est')['open'].first()
    df['session_open_price'] = df['date_est'].map(session_open)

    # Compute previous bar high/low for shift detection
    prev_high = df['high'].shift(1)
    prev_low = df['low'].shift(1)

    # Bullish shift: higher high and higher low
    bullish_shift = (df['high'] > prev_high) & (df['low'] > prev_low)
    # Bearish shift: lower high and lower low
    bearish_shift = (df['low'] < prev_low) & (df['high'] < prev_high)

    # Entry conditions
    long_entry = bullish_shift & (df['close'] > df['session_open_price'])
    short_entry = bearish_shift & (df['close'] < df['session_open_price'])

    # Store as columns for easy access
    df['long_entry'] = long_entry
    df['short_entry'] = short_entry

    entries = []
    trade_num = 1

    for i in range(len(df)):
        # Skip rows with NaN session open price (e.g., first day if missing)
        if pd.isna(df['session_open_price'].iloc[i]):
            continue
        # Long entry
        if df['long_entry'].iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        # Short entry
        elif df['short_entry'].iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries