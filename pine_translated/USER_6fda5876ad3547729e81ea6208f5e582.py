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
    entries = []
    trade_num = 1

    n = len(df)
    if n < 3:
        return entries

    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    time = df['time'].values

    # State variables
    pd_high = np.nan
    pd_low = np.nan
    temp_high = np.nan
    temp_low = np.nan
    swept_high = False
    swept_low = False
    ig_active = False
    ig_direction = 0
    ig_c1_high = np.nan
    ig_c1_low = np.nan
    ig_validation_end = -1

    # London trading windows: 07:00-11:45 and 14:00-14:45
    def in_london_window(ts):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        time_val = hour * 60 + minute
        window1_start = 7 * 60
        window1_end = 11 * 60 + 45
        window2_start = 14 * 60
        window2_end = 14 * 60 + 45
        return (window1_start <= time_val <= window1_end) or (window2_start <= time_val <= window2_end)

    for i in range(n):
        current_time = time[i]
        current_high = high[i]
        current_low = low[i]
        current_close = close[i]

        # Detect new day
        is_new_day = False
        if i > 0:
            prev_dt = datetime.fromtimestamp(time[i-1], tz=timezone.utc)
            curr_dt = datetime.fromtimestamp(current_time, tz=timezone.utc)
            if prev_dt.date() < curr_dt.date():
                is_new_day = True
        elif i == 0:
            is_new_day = True

        # Update PDH and PDL on new day
        if is_new_day:
            pd_high = temp_high
            pd_low = temp_low
            temp_high = current_high
            temp_low = current_low
            swept_high = False
            swept_low = False
        else:
            if np.isnan(temp_high) or current_high > temp_high:
                temp_high = current_high
            if np.isnan(temp_low) or current_low < temp_low:
                temp_low = current_low

        # Sweep detection (once per day)
        sweep_high_now = not swept_high and not np.isnan(pd_high) and current_high > pd_high
        sweep_low_now = not swept_low and not np.isnan(pd_low) and current_low < pd_low

        if sweep_high_now:
            swept_high = True

        if sweep_low_now:
            swept_low = True

        # Detect FVG (requires at least 3 bars: bar[i-2], bar[i-1], bar[i])
        if i >= 2:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev2_high = high[i-2]
            prev2_low = low[i-2]

            bullish_fvg = prev2_high < current_low
            bearish_fvg = prev2_low > current_high

            if bullish_fvg:
                ig_active = True
                ig_direction = 1
                ig_c1_high = prev2_high
                ig_c1_low = prev2_low
                ig_validation_end = i + 4

            if bearish_fvg:
                ig_active = True
                ig_direction = -1
                ig_c1_high = prev2_high
                ig_c1_low = prev2_low
                ig_validation_end = i + 4

        # Validation check
        validated = False
        if ig_active and i <= ig_validation_end:
            if ig_direction == 1 and current_close < ig_c1_high:
                validated = True
            if ig_direction == -1 and current_close > ig_c1_low:
                validated = True

        # Entry conditions
        in_window = in_london_window(current_time)
        system_entry_long = validated and ig_direction == -1 and sweep_low_now and in_window
        system_entry_short = validated and ig_direction == 1 and sweep_high_now and in_window

        # Execute entries
        if system_entry_long:
            entry_price = current_close
            entry_ts = int(current_time)
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
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
            ig_active = False

        if system_entry_short:
            entry_price = current_close
            entry_ts = int(current_time)
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
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
            ig_active = False

        # Clear validation flag
        if validated:
            ig_active = False

    return entries