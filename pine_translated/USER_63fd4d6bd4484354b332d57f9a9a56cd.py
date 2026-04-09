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
    results = []
    trade_num = 1

    if len(df) < 5:
        return results

    high = df['high']
    low = df['low']
    close = df['close']
    time = df['time']

    # Detect new day
    df['day'] = pd.to_datetime(df['time'], unit='s').dt.date
    new_day = df['day'].diff().fillna(0) != 0

    # Calculate Previous Day High and Low
    pd_high = np.nan
    pd_low = np.nan
    temp_high = np.nan
    temp_low = np.nan

    pd_high_arr = np.full(len(df), np.nan)
    pd_low_arr = np.full(len(df), np.nan)

    for i in range(len(df)):
        if new_day.iloc[i]:
            pd_high = temp_high if not np.isnan(temp_high) else pd_high
            pd_low = temp_low if not np.isnan(temp_low) else pd_low
            pd_high_arr[i] = pd_high
            pd_low_arr[i] = pd_low
            temp_high = high.iloc[i]
            temp_low = low.iloc[i]
        else:
            temp_high = high.iloc[i] if np.isnan(temp_high) else max(temp_high, high.iloc[i])
            temp_low = low.iloc[i] if np.isnan(temp_low) else min(temp_low, low.iloc[i])
            pd_high_arr[i] = pd_high
            pd_low_arr[i] = pd_low

    # Sweep detection
    swept_high = False
    swept_low = False

    for i in range(1, len(df)):
        if new_day.iloc[i]:
            swept_high = False
            swept_low = False

        if not swept_high and high.iloc[i] > pd_high_arr[i] and pd_high_arr[i] > 0:
            swept_high = True
            # LONG entry on PDH sweep
            entry_ts = int(time.iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = close.iloc[i]
            results.append({
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

        if not swept_low and low.iloc[i] < pd_low_arr[i] and pd_low_arr[i] > 0:
            swept_low = True
            # SHORT entry on PDL sweep
            entry_ts = int(time.iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = close.iloc[i]
            results.append({
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

        if swept_high and swept_low:
            swept_high = False
            swept_low = False

    return results