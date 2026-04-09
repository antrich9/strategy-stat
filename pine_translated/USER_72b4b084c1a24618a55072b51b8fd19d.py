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
    # Convert Unix timestamps to datetime (UTC) and then to Europe/London for session detection
    ts = pd.to_datetime(df['time'], unit='s', utc=True)
    ts_london = ts.dt.tz_convert('Europe/London')

    # Day of month in London time (used to detect a new day)
    day_london = ts_london.dt.day

    # Minute of day in London time
    hour = ts_london.dt.hour
    minute = ts_london.dt.minute
    minute_of_day = hour * 60 + minute

    # Asia session window: 07:45 (inclusive) to 11:45 (exclusive)
    asia_start = 7 * 60 + 45   # 465
    asia_end   = 11 * 60 + 45  # 705
    in_asia = (minute_of_day >= asia_start) & (minute_of_day < asia_end)

    # Price arrays
    high_arr   = df['high'].values
    low_arr    = df['low'].values
    close_arr  = df['close'].values
    time_arr   = df['time'].values

    entries = []
    trade_num = 1

    # Session tracking variables
    session_high = np.nan
    session_low  = np.nan
    session_broken_flag = False
    last_day = -1

    for i in range(len(df)):
        cur_day = day_london.iloc[i]

        # Reset session variables at the start of each new London day
        if cur_day != last_day:
            session_high = np.nan
            session_low  = np.nan
            session_broken_flag = False
            last_day = cur_day

        # Update session high/low while inside the Asia session
        if in_asia.iloc[i]:
            if np.isnan(session_high):
                session_high = high_arr[i]
            else:
                session_high = max(session_high, high_arr[i])

            if np.isnan(session_low):
                session_low = low_arr[i]
            else:
                session_low = min(session_low, low_arr[i])

        # After the Asia session ends, check for breaks (only once per day)
        if not in_asia.iloc[i] and not session_broken_flag:
            if not np.isnan(session_high) and not np.isnan(session_low):
                long_trigger  = high_arr[i] > session_high
                short_trigger = low_arr[i] < session_low

                if long_trigger:
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'long',
                        'entry_ts': int(time_arr[i]),
                        'entry_time': datetime.fromtimestamp(time_arr[i], tz=timezone.utc).isoformat(),
                        'entry_price_guess': float(close_arr[i]),
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': float(close_arr[i]),
                        'raw_price_b': float(close_arr[i])
                    })
                    trade_num += 1

                if short_trigger:
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'short',
                        'entry_ts': int(time_arr[i]),
                        'entry_time': datetime.fromtimestamp(time_arr[i], tz=timezone.utc).isoformat(),
                        'entry_price_guess': float(close_arr[i]),
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': float(close_arr[i]),
                        'raw_price_b': float(close_arr[i])
                    })
                    trade_num += 1

                if long_trigger or short_trigger:
                    session_broken_flag = True

    return entries