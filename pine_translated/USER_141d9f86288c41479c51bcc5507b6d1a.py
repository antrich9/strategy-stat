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
    close = df['close']
    high = df['high']
    low = df['low']
    open_ = df['open']
    volume = df['volume']
    time_col = df['time']

    # ATR Filter (Wilder)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean()
    atrfilt_bull = (low - high.shift(2) > atr / 1.5)
    atrfilt_bear = (high - low.shift(2) > atr / 1.5) | (low.shift(2) - high > atr / 1.5)

    # Volume Filter
    volfilt = volume.shift(1) > volume.rolling(9).mean() * 1.5

    # Trend Filter (54 SMA)
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2

    # Check if current bar is in specified time windows (0700-0959 or 1200-1459)
    def is_in_time_windows(ts):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        time_val = hour * 100 + minute
        return (700 <= time_val <= 959) or (1200 <= time_val <= 1459)

    time_filter = time_col.apply(is_in_time_windows)

    # Detect new days and compute prev day high/low
    dt_times = pd.to_datetime(time_col, unit='s', utc=True)
    is_new_day = dt_times.dt.date != dt_times.dt.date.shift(1)
    is_new_day.iloc[0] = True

    prev_day_high = high.rolling(window=2).apply(lambda x: x.iloc[0] if len(x) > 0 else np.nan)
    prev_day_low = low.rolling(window=2).apply(lambda x: x.iloc[0] if len(x) > 0 else np.nan)

    # For each bar, we need prev day H/L from previous day
    prev_day_high_arr = np.full(len(df), np.nan)
    prev_day_low_arr = np.full(len(df), np.nan)

    daily_highs = []
    daily_lows = []
    daily_starts = []

    for i in range(len(df)):
        dt = datetime.fromtimestamp(time_col.iloc[i], tz=timezone.utc)
        day_start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        if i == 0:
            daily_starts.append(day_start)
            daily_highs.append(high.iloc[i])
            daily_lows.append(low.iloc[i])
        else:
            prev_dt = datetime.fromtimestamp(time_col.iloc[i-1], tz=timezone.utc)
            if dt.date() != prev_dt.date():
                prev_day_high_arr[i] = daily_highs[-1]
                prev_day_low_arr[i] = daily_lows[-1]
                daily_starts.append(day_start)
                daily_highs = [high.iloc[i]]
                daily_lows = [low.iloc[i]]
            else:
                daily_highs[-1] = max(daily_highs[-1], high.iloc[i])
                daily_lows[-1] = min(daily_lows[-1], low.iloc[i])
                if i > 0:
                    prev_day_high_arr[i] = prev_day_high_arr[i-1]
                    prev_day_low_arr[i] = prev_day_low_arr[i-1]

    prev_day_high = pd.Series(prev_day_high_arr, index=df.index)
    prev_day_low = pd.Series(prev_day_low_arr, index=df.index)

    # Flag detection: flagpdh = close > prev day high, flagpdl = close < prev day low
    flagpdh = close > prev_day_high
    flagpdl = close < prev_day_low

    # Reset flags on new days
    for i in range(1, len(df)):
        if is_new_day.iloc[i]:
            flagpdh.iloc[i] = False
            flagpdl.iloc[i] = False

    # OB/FVG conditions
    def is_up(idx):
        return close.iloc[idx] > open_.iloc[idx]

    def is_down(idx):
        return close.iloc[idx] < open_.iloc[idx]

    ob_up = is_down(1) & is_up(0) & (close > high.shift(1))
    ob_down = is_up(1) & is_down(0) & (close < low.shift(1))
    fvg_up = low > high.shift(2)
    fvg_down = high < low.shift(2)

    stacked_bull = ob_up & fvg_up
    stacked_bear = ob_down & fvg_down

    # Long entry condition: flagpdh swept, bull stacked OB+FVG, filters
    long_cond = (flagpdh.shift(1)) & stacked_bull & volfilt & atrfilt_bull & locfiltb & time_filter

    # Short entry condition: flagpdl swept, bear stacked OB+FVG, filters
    short_cond = (flagpdl.shift(1)) & stacked_bear & volfilt & atrfilt_bear & locfilts & time_filter

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if pd.isna(prev_day_high.iloc[i]) or pd.isna(prev_day_low.iloc[i]):
            continue

        if long_cond.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(time_col.iloc[i]),
                'entry_time': datetime.fromtimestamp(time_col.iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1

        if short_cond.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(time_col.iloc[i]),
                'entry_time': datetime.fromtimestamp(time_col.iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1

    return entries