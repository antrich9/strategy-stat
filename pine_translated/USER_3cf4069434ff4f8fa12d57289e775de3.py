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
    open_col = df['open']
    high_col = df['high']
    low_col = df['low']
    close_col = df['close']
    volume_col = df['volume']
    time_col = df['time']

    # Time window filter (London time: 8:00-9:45 and 15:00-16:45)
    hour = pd.to_datetime(time_col, unit='ms', utc=True).dt.hour
    minute = pd.to_datetime(time_col, unit='ms', utc=True).dt.minute
    time_window = ((hour == 8) | ((hour == 9) & (minute <= 45)) |
                   (hour == 15) | ((hour == 16) & (minute <= 45)))

    # Bullish OB: isDown(2) and isUp(1) and close[1] > high[2]
    is_down_2 = close_col.shift(2) < open_col.shift(2)
    is_up_1 = close_col.shift(1) > open_col.shift(1)
    ob_up = is_down_2 & is_up_1 & (close_col.shift(1) > high_col.shift(2))

    # Bearish OB: isUp(2) and isDown(1) and close[1] < low[2]
    is_up_2 = close_col.shift(2) > open_col.shift(2)
    is_down_1 = close_col.shift(1) < open_col.shift(1)
    ob_down = is_up_2 & is_down_1 & (close_col.shift(1) < low_col.shift(2))

    # Bullish FVG: low > high[2]
    high_shifted_2 = high_col.shift(2)
    fvg_up = low_col > high_shifted_2

    # Bearish FVG: high < low[2]
    low_shifted_2 = low_col.shift(2)
    fvg_down = high_col < low_shifted_2

    # Entry conditions
    long_entry = ob_up & fvg_up & time_window
    short_entry = ob_down & fvg_down & time_window

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if i < 2:
            continue
        if pd.isna(close_col.iloc[i]) or pd.isna(close_col.iloc[i-1]) or pd.isna(close_col.iloc[i-2]):
            continue
        if pd.isna(low_col.iloc[i]) or pd.isna(high_col.iloc[i]) or pd.isna(high_shifted_2.iloc[i]) or pd.isna(low_shifted_2.iloc[i]):
            continue

        ts = int(time_col.iloc[i])
        entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
        entry_price = float(close_col.iloc[i])

        if long_entry.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif short_entry.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
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