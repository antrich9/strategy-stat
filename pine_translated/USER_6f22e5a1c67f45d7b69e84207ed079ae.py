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

    close = df['close']
    open_price = df['open']
    high = df['high']
    low = df['low']
    volume = df['volume']
    time = df['time']

    n = len(df)

    def is_up(idx):
        return close.iloc[idx] > open_price.iloc[idx]

    def is_down(idx):
        return close.iloc[idx] < open_price.iloc[idx]

    def is_ob_up(idx):
        return is_down(idx + 1) and is_up(idx) and close.iloc[idx] > high.iloc[idx + 1]

    def is_ob_down(idx):
        return is_up(idx + 1) and is_down(idx) and close.iloc[idx] < low.iloc[idx + 1]

    def is_fvg_up(idx):
        return low.iloc[idx] > high.iloc[idx + 2]

    def is_fvg_down(idx):
        return high.iloc[idx] < low.iloc[idx + 2]

    ob_up = pd.Series(False, index=df.index)
    ob_down = pd.Series(False, index=df.index)
    fvg_up = pd.Series(False, index=df.index)
    fvg_down = pd.Series(False, index=df.index)

    for i in range(2, n):
        ob_up.iloc[i] = is_ob_up(1)
        ob_down.iloc[i] = is_ob_down(1)
        fvg_up.iloc[i] = is_fvg_up(0)
        fvg_down.iloc[i] = is_fvg_down(0)

    volfilt = volume.rolling(9).mean() * 1.5 < volume

    delta_high = high.diff(2)
    delta_low = -low.diff(2)
    tr = pd.concat([delta_high, delta_low, (high - low).shift(1), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean()

    loc = close.ewm(span=54, adjust=False).mean()
    loc2 = loc > loc.shift(1)

    atrfilt_long = (low - high.shift(2) > atr) | (low.shift(2) - high > atr)
    atrfilt_short = (low - high.shift(2) > atr) | (low.shift(2) - high > atr)

    bfvg = (low > high.shift(2)) & volfilt & atrfilt_long & loc2
    sfvg = (high < low.shift(2)) & volfilt & atrfilt_short & (~loc2)

    london_tz_offset = 0

    in_morning_window = pd.Series(False, index=df.index)
    in_afternoon_window = pd.Series(False, index=df.index)

    for i in range(n):
        ts = datetime.fromtimestamp(time.iloc[i], tz=timezone.utc)
        hour = ts.hour
        minute = ts.minute

        morning_start_hour, morning_start_min = 7, 45
        morning_end_hour, morning_end_min = 9, 45
        afternoon_start_hour, afternoon_start_min = 14, 45
        afternoon_end_hour, afternoon_end_min = 16, 45

        in_morning = (hour > morning_start_hour or (hour == morning_start_hour and minute >= morning_start_min)) and \
                     (hour < morning_end_hour or (hour == morning_end_hour and minute < morning_end_min))
        in_afternoon = (hour > afternoon_start_hour or (hour == afternoon_start_hour and minute >= afternoon_start_min)) and \
                       (hour < afternoon_end_hour or (hour == afternoon_end_hour and minute < afternoon_end_min))

        in_morning_window.iloc[i] = in_morning
        in_afternoon_window.iloc[i] = in_afternoon

    in_trading_window = in_morning_window | in_afternoon_window

    long_entry = ob_up & fvg_up & in_trading_window
    short_entry = ob_down & fvg_down & in_trading_window

    for i in range(n):
        if i < 2:
            continue

        direction = None
        if long_entry.iloc[i]:
            direction = 'long'
        elif short_entry.iloc[i]:
            direction = 'short'

        if direction:
            entry_price = close.iloc[i]
            entry_ts = int(time.iloc[i])
            entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()

            results.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return results