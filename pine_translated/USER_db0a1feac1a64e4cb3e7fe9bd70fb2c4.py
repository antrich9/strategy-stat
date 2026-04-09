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
    if len(df) < 5:
        return []

    results = []
    trade_num = 1

    # Asia session: 2300-0700 London time
    london_offset = 1  # GMT+1

    def in_asia_session(ts):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        utc_hour = dt.hour
        london_hour = (utc_hour - london_offset) % 24
        return (0 <= london_hour < 8)

    def in_time_filter(ts):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        utc_hour = dt.hour
        london_hour = (utc_hour - london_offset) % 24
        time_val = london_hour * 100 + dt.minute
        in_window1 = (700 <= time_val <= 959)
        in_window2 = (1200 <= time_val <= 1459)
        return in_window1 or in_window2

    # Resample to Daily for previous day high/low
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    daily = df.groupby(df['datetime'].dt.date).agg({'high': 'max', 'low': 'min'}).reset_index()
    daily['datetime'] = pd.to_datetime(daily['datetime'])
    daily['prev_day_high'] = daily['high'].shift(1)
    daily['prev_day_low'] = daily['low'].shift(1)
    daily = daily.dropna(subset=['prev_day_high', 'prev_day_low'])

    prev_day_high = df['time'].map(lambda x: daily[daily['datetime'] <= pd.Timestamp(datetime.fromtimestamp(x, tz=timezone.utc)).date()]['prev_day_high'].iloc[-1] if len(daily[daily['datetime'] <= pd.Timestamp(datetime.fromtimestamp(x, tz=timezone.utc)).date()]) > 0 else np.nan)
    prev_day_low = df['time'].map(lambda x: daily[daily['datetime'] <= pd.Timestamp(datetime.fromtimestamp(x, tz=timezone.utc)).date()]['prev_day_low'].iloc[-1] if len(daily[daily['datetime'] <= pd.Timestamp(datetime.fromtimestamp(x, tz=timezone.utc)).date()]) > 0 else np.nan)

    prev_day_high_filled = prev_day_high.ffill()
    prev_day_low_filled = prev_day_low.ffill()

    # Sweep detection
    pdh_swept = df['high'] > prev_day_high_filled
    pdl_swept = df['low'] < prev_day_low_filled

    # Asia session high/low
    asia_high = pd.Series(np.nan, index=df.index)
    asia_low = pd.Series(np.nan, index=df.index)
    session_started = False
    curr_asia_high = np.nan
    curr_asia_low = np.nan

    for i in df.index:
        if in_asia_session(df['time'].iloc[i]):
            if not session_started:
                session_started = True
                curr_asia_high = df['high'].iloc[i]
                curr_asia_low = df['low'].iloc[i]
            else:
                curr_asia_high = max(curr_asia_high, df['high'].iloc[i])
                curr_asia_low = min(curr_asia_low, df['low'].iloc[i])
            asia_high.iloc[i] = curr_asia_high
            asia_low.iloc[i] = curr_asia_low
        else:
            session_started = False

    asia_high_prev = asia_high.shift(1)
    asia_low_prev = asia_low.shift(1)
    asia_high_swept = df['high'] > asia_high_prev
    asia_low_swept = df['low'] < asia_low_prev

    # OB and FVG detection
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']

    ob_up = pd.Series(False, index=df.index)
    ob_down = pd.Series(False, index=df.index)
    fvg_up = pd.Series(False, index=df.index)
    fvg_down = pd.Series(False, index=df.index)

    for i in range(1, len(df) - 2):
        ob_up.iloc[i] = (close.iloc[i] > open_.iloc[i]) and (close.iloc[i-1] < open_.iloc[i-1]) and (close.iloc[i] > high.iloc[i-1])
        ob_down.iloc[i] = (close.iloc[i] < open_.iloc[i]) and (close.iloc[i-1] > open_.iloc[i-1]) and (close.iloc[i] < low.iloc[i-1])
        fvg_up.iloc[i] = low.iloc[i] > high.iloc[i + 2]
        fvg_down.iloc[i] = high.iloc[i] < low.iloc[i + 2]

    # Session state
    flagpdh = False
    flagpdl = False
    waiting_for_entry_long = False
    waiting_for_entry_short = False

    for i in range(1, len(df)):
        if pd.isna(prev_day_high_filled.iloc[i]) or pd.isna(prev_day_low_filled.iloc[i]):
            continue

        # Reset on new day
        if i > 1:
            curr_date = datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).date()
            prev_date = datetime.fromtimestamp(df['time'].iloc[i-1], tz=timezone.utc).date()
            if curr_date != prev_date:
                flagpdh = False
                flagpdl = False
                waiting_for_entry_long = False
                waiting_for_entry_short = False

        # Sweep flags
        if pdh_swept.iloc[i]:
            flagpdh = True
        if pdl_swept.iloc[i]:
            flagpdl = True

        # Time filter
        if not in_time_filter(df['time'].iloc[i]):
            continue

        # Entry logic - Long
        if flagpdl and ob_up.iloc[i] and fvg_up.iloc[i]:
            entry_price = close.iloc[i]
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1

        # Entry logic - Short
        if flagpdh and ob_down.iloc[i] and fvg_down.iloc[i]:
            entry_price = close.iloc[i]
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1

    return results