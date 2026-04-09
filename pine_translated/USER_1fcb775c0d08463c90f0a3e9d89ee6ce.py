import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

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
    if len(df) < 5:
        return []

    entries = []
    trade_num = 1
    in_position = False
    in_short_position = False

    # Trading window parameters
    start_hour_1, end_hour_1, end_minute_1 = 7, 10, 59
    start_hour_2, end_hour_2, end_minute_2 = 15, 16, 59

    prev_day_high = np.nan
    prev_day_low = np.nan
    flagpdh = False
    flagpdl = False
    prev_day_id = -1

    # Detect FVG conditions for additional entries
    bull_fvg = (df['low'].shift(2) > df['high'].shift(1)) & \
               (df['low'] > df['high'].shift(1)) & \
               (df['high'].shift(2) < df['low'].shift(1))
    bear_fvg = (df['high'].shift(2) < df['low'].shift(1)) & \
               (df['high'] < df['low'].shift(1)) & \
               (df['low'].shift(2) > df['high'].shift(1))

    for i in range(1, len(df)):
        ts = df['time'].iloc[i]
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        day_id = ts // 86400000

        # DST calculation (US DST rules)
        year = dt.year
        march_1 = datetime(year, 3, 1)
        first_sunday_march = 7 - march_1.weekday()
        if first_sunday_march <= 0:
            first_sunday_march += 7
        second_sunday_march = first_sunday_march + 7
        november_1 = datetime(year, 11, 1)
        first_sunday_november = 7 - november_1.weekday()
        if first_sunday_november <= 0:
            first_sunday_november += 7
        dst_start = datetime(year, 3, second_sunday_march)
        dst_end = datetime(year, 11, first_sunday_november)
        is_dst = dst_start <= dt.replace(month=dt.month, day=dt.day) < dst_end

        adjusted_timezone_offset = 3600000 if is_dst else 0
        adjusted_time = dt + timedelta(milliseconds=adjusted_timezone_offset)
        adjusted_hour = adjusted_time.hour
        adjusted_minute = adjusted_time.minute

        # Trading window check
        in_window_1 = (adjusted_hour > start_hour_1 and adjusted_hour <= end_hour_1) or \
                      (adjusted_hour == end_hour_1 and adjusted_minute <= end_minute_1)
        in_window_2 = (adjusted_hour >= start_hour_2 and adjusted_hour <= end_hour_2) or \
                      (adjusted_hour == end_hour_2 and adjusted_minute <= end_minute_2)
        in_trading_window = in_window_1 or in_window_2

        # New day detection
        if day_id != prev_day_id:
            prev_day_data = df.iloc[max(0, i - 1440):i]
            if len(prev_day_data) > 0:
                prev_day_high = prev_day_data['high'].max()
                prev_day_low = prev_day_data['low'].min()

            flagpdh = False
            flagpdl = False
            prev_day_id = day_id
            in_position = False
            in_short_position = False

        # Current day high/low (from day start to current bar)
        current_day_data = df.iloc[max(0, i - 240):i + 1]
        if len(current_day_data) > 0:
            current_day_high = current_day_data['high'].max()
            current_day_low = current_day_data['low'].min()
        else:
            current_day_high, current_day_low = np.nan, np.nan

        # Previous day high/low taken conditions
        if not np.isnan(prev_day_high):
            prev_day_high_taken = df['high'].iloc[i] > prev_day_high
            if prev_day_high_taken and current_day_low > prev_day_low:
                flagpdh = True

        if not np.isnan(prev_day_low):
            prev_day_low_taken = df['low'].iloc[i] < prev_day_low
            if prev_day_low_taken and current_day_high < prev_day_high:
                flagpdl = True

        # Entry conditions based on flagpdh/flagpdl
        if in_trading_window:
            if flagpdh and not in_position and not in_short_position:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(df['close'].iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(df['close'].iloc[i]),
                    'raw_price_b': float(df['close'].iloc[i])
                })
                trade_num += 1
                in_position = True

            elif flagpdl and not in_short_position and not in_position:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(df['close'].iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(df['close'].iloc[i]),
                    'raw_price_b': float(df['close'].iloc[i])
                })
                trade_num += 1
                in_short_position = True

    return entries