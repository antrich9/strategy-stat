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
    if n < 5:
        return entries

    # Previous day high/low - use daily high/low shifted
    prev_day_high = df['high'].rolling(window=2).max().shift(1)
    prev_day_low = df['low'].rolling(window=2).min().shift(1)

    # Flags for previous day high/low sweep
    flagpdh = (df['close'] > prev_day_high).astype(int)
    flagpdl = (df['close'] < prev_day_low).astype(int)

    # Cumulative sweep flags (once swept, stays true until new day)
    flagpdh_cum = flagpdh.cumsum()
    flagpdl_cum = flagpdl.cumsum()

    # Detect new day (previous close != today's close pattern or time jump)
    # Use day change detection
    times = df['time']
    day_num = (times // 86400000).astype(int)
    is_new_day = day_num.diff().fillna(0) != 0

    # Reset flags on new day
    flagpdh_swept = pd.Series(False, index=df.index)
    flagpdl_swept = pd.Series(False, index=df.index)

    current_pdh = False
    current_pdl = False
    for i in range(n):
        if is_new_day.iloc[i]:
            current_pdh = False
            current_pdl = False
        if flagpdh.iloc[i]:
            current_pdh = True
        if flagpdl.iloc[i]:
            current_pdl = True
        flagpdh_swept.iloc[i] = current_pdh
        flagpdl_swept.iloc[i] = current_pdl

    # OB and FVG conditions
    # isUp(index): close[index] > open[index]
    # isDown(index): close[index] < open[index]
    # isObUp(index): isDown(index + 1) and isUp(index) and close[index] > high[index + 1]
    # isObDown(index): isUp(index + 1) and isDown(index) and close[index] < low[index + 1]
    # isFvgUp(index): low[index] > high[index + 2]
    # isFvgDown(index): high[index] < low[index + 2]

    is_up = df['close'] > df['open']
    is_down = df['close'] < df['open']

    ob_up = is_down.shift(1) & is_up & (df['close'] > df['high'].shift(1))
    ob_down = is_up.shift(1) & is_down & (df['close'] < df['low'].shift(1))

    fvg_up = df['low'] > df['high'].shift(2)
    fvg_down = df['high'] < df['low'].shift(2)

    # Stacked OB + FVG conditions
    stacked_bullish = ob_up.shift(1) & fvg_up
    stacked_bearish = ob_down.shift(1) & fvg_down

    # Time filter - need to convert times to hour/minute
    # times are unix timestamps in milliseconds
    times_dt = pd.to_datetime(df['time'], unit='ms', utc=True)
    hours = times_dt.dt.hour
    minutes = times_dt.dt.minute
    total_minutes = hours * 60 + minutes

    # Long entry time window: 0700-0959
    in_long_window = (total_minutes >= 420) & (total_minutes <= 599)
    # Short entry time window: 1200-1459
    in_short_window = (total_minutes >= 720) & (total_minutes <= 899)

    # Waiting flags - var bool, reset on new day
    waiting_for_entry = False
    waiting_for_short = False

    for i in range(n):
        if is_new_day.iloc[i]:
            waiting_for_entry = False
            waiting_for_short = False

        # Skip if any required indicator is NaN
        if i < 2:
            continue
        if pd.isna(ob_up.iloc[i]) or pd.isna(fvg_up.iloc[i]) or pd.isna(ob_down.iloc[i]) or pd.isna(fvg_down.iloc[i]):
            continue
        if pd.isna(flagpdh_swept.iloc[i]) or pd.isna(flagpdl_swept.iloc[i]):
            continue

        # LONG entry conditions
        if (flagpdh_swept.iloc[i] and
            not waiting_for_entry and
            in_long_window.iloc[i] and
            stacked_bullish.iloc[i]):

            entry_ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])

            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })

            trade_num += 1
            waiting_for_entry = True

        # SHORT entry conditions
        if (flagpdl_swept.iloc[i] and
            not waiting_for_short and
            in_short_window.iloc[i] and
            stacked_bearish.iloc[i]):

            entry_ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])

            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })

            trade_num += 1
            waiting_for_short = True

    return entries