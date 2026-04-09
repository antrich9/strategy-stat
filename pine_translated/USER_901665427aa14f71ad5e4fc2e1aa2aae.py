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
    # Time window: 7:45-9:45 and 14:45-16:45 UTC
    morning_start = datetime(2020, 1, 1, 7, 45, tzinfo=timezone.utc)
    morning_end = datetime(2020, 1, 1, 9, 45, tzinfo=timezone.utc)
    afternoon_start = datetime(2020, 1, 1, 14, 45, tzinfo=timezone.utc)
    afternoon_end = datetime(2020, 1, 1, 16, 45, tzinfo=timezone.utc)

    ts = pd.to_datetime(df['time'], unit='s', utc=True)
    in_morning = (ts.dt.hour == 7) & (ts.dt.minute >= 45) | (ts.dt.hour == 8) | (ts.dt.hour == 9) & (ts.dt.minute < 45)
    in_afternoon = (ts.dt.hour == 14) & (ts.dt.minute >= 45) | (ts.dt.hour == 15) | (ts.dt.hour == 16) & (ts.dt.minute < 45)
    in_time_window = in_morning | in_afternoon

    # Filters (all disabled by default inp1=inp2=inp3=false)
    volfilt = True
    atrfilt = True
    locfiltb = True
    locfilts = True

    # OB conditions
    is_up = df['close'] > df['open']
    is_down = df['close'] < df['open']
    ob_up = is_up & is_down.shift(1) & (df['close'] > df['high'].shift(1))
    ob_down = is_down & is_up.shift(1) & (df['close'] < df['low'].shift(1))

    # FVG conditions
    fvg_up = df['low'] > df['high'].shift(2)
    fvg_down = df['high'] < df['low'].shift(2)

    # Entry conditions
    long_entry = ob_up & fvg_up & in_time_window
    short_entry = ob_down & fvg_down & in_time_window

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if long_entry.iloc[i]:
            ts_val = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts_val,
                'entry_time': datetime.fromtimestamp(ts_val, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif short_entry.iloc[i]:
            ts_val = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts_val,
                'entry_time': datetime.fromtimestamp(ts_val, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1

    return entries