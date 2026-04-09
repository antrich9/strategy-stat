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
    if len(df) < 20:
        return []

    times = df['time'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    prev_day_high = np.full(len(df), np.nan)
    prev_day_low = np.full(len(df), np.nan)
    is_new_day = np.zeros(len(df), dtype=bool)

    for i in range(1, len(df)):
        current_date = datetime.fromtimestamp(times[i], tz=timezone.utc).date()
        prev_date = datetime.fromtimestamp(times[i-1], tz=timezone.utc).date()
        is_new_day[i] = current_date > prev_date
        if is_new_day[i]:
            prev_day_high[i] = highs[i-1]
            prev_day_low[i] = lows[i-1]

    for i in range(1, len(df)):
        if np.isnan(prev_day_high[i]):
            prev_day_high[i] = prev_day_high[i-1]
        if np.isnan(prev_day_low[i]):
            prev_day_low[i] = prev_day_low[i-1]

    flagpdl = False
    flagpdh = False
    waiting_for_entry = False
    waiting_for_short_entry = False
    in_long = False
    in_short = False
    entries = []
    trade_num = 1

    def is_up(idx):
        return closes[idx] > opens[idx]

    def is_down(idx):
        return closes[idx] < opens[idx]

    def is_ob_up(idx):
        return is_down(idx + 1) and is_up(idx) and closes[idx] > highs[idx + 1]

    def is_ob_down(idx):
        return is_up(idx + 1) and is_down(idx) and closes[idx] < lows[idx + 1]

    def is_fvg_up(idx):
        return lows[idx] > highs[idx + 2]

    def is_fvg_down(idx):
        return highs[idx] < lows[idx + 2]

    for i in range(2, len(df)):
        ts = int(times[i])

        if is_new_day[i]:
            flagpdl = False
            flagpdh = False
            waiting_for_entry = False
            waiting_for_short_entry = False
            in_long = False
            in_short = False

        if closes[i] > prev_day_high[i]:
            flagpdh = True
        if closes[i] < prev_day_low[i]:
            flagpdl = True

        ob_up = is_ob_up(1)
        ob_down = is_ob_down(1)
        fvg_up = is_fvg_up(0)
        fvg_down = is_fvg_down(0)

        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        current_time_mins = hour * 60 + minute
        in_long_window = 7 * 60 <= current_time_mins <= 9 * 60 + 59
        in_short_window = 12 * 60 <= current_time_mins <= 14 * 60 + 59

        if flagpdl and not in_long and ob_up and fvg_up and in_long_window:
            entry_ts = int(times[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(closes[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(closes[i]),
                'raw_price_b': float(closes[i])
            })
            trade_num += 1
            in_long = True

        if flagpdh and not in_short and ob_down and fvg_down and in_short_window:
            entry_ts = int(times[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(closes[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(closes[i]),
                'raw_price_b': float(closes[i])
            })
            trade_num += 1
            in_short = True

    return entries