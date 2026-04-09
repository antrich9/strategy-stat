import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    time_vals = df['time'].values
    open_vals = df['open'].values
    high_vals = df['high'].values
    low_vals = df['low'].values
    close_vals = df['close'].values

    open_s = pd.Series(open_vals)
    high_s = pd.Series(high_vals)
    low_s = pd.Series(low_vals)
    close_s = pd.Series(close_vals)

    # Identify daily boundaries using time column
    dt_series = pd.to_datetime(pd.Series(time_vals), unit='s', utc=True)
    day_change = dt_series.dt.date != dt_series.dt.date.shift(1)
    day_change.iloc[0] = True

    # Previous day high/low: for each bar, find the high/low of the previous completed day
    prev_day_high = high_s.where(day_change).ffill().shift(1)
    prev_day_low = low_s.where(day_change).ffill().shift(1)

    # Detect sweeps of previous day high/low
    pdh_swept = close_s > prev_day_high
    pdl_swept = close_s < prev_day_low

    # OB detection
    is_up = close_s > open_s
    is_down = close_s < open_s

    ob_up = is_down.shift(1) & is_up & (close_s > high_s.shift(1))
    ob_down = is_up.shift(1) & is_down & (close_s < low_s.shift(1))

    # FVG detection
    fvg_up = low_s > high_s.shift(2)
    fvg_down = high_s < low_s.shift(2)

    # Stacked OB + FVG conditions
    bullish_stack = ob_up & fvg_up
    bearish_stack = ob_down & fvg_down

    # Time filter: 0700-0959 and 1200-1459 UTC
    hour = dt_series.dt.hour
    time_cond1 = (hour >= 7) & (hour < 10)
    time_cond2 = (hour >= 12) & (hour < 15)
    time_filter = time_cond1 | time_cond2

    # Build boolean series for entry conditions
    long_entry_cond = pdh_swept & bullish_stack & time_filter
    short_entry_cond = pdl_swept & bearish_stack & time_filter

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if i < 3:
            continue
        if pd.isna(prev_day_high.iloc[i]) or pd.isna(prev_day_low.iloc[i]):
            continue
        if pd.isna(bullish_stack.iloc[i]) or pd.isna(bearish_stack.iloc[i]):
            continue
        if pd.isna(time_filter.iloc[i]):
            continue

        direction = None
        if long_entry_cond.iloc[i]:
            direction = 'long'
        elif short_entry_cond.iloc[i]:
            direction = 'short'

        if direction:
            ts = int(time_vals[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(close_vals[i])

            entries.append({
                'trade_num': trade_num,
                'direction': direction,
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