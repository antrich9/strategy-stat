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
    # Candle direction
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']
    volume = df['volume']

    is_up = close > open_
    is_down = close < open_

    # Order Block (OB) detection
    # isObUp(index) => isDown(index+1) & isUp(index) & close[index] > high[index+1]
    is_ob_up = is_down.shift(2) & is_up.shift(1) & (close.shift(1) > high.shift(2))
    # isObDown(index) => isUp(index+1) & isDown(index) & close[index] < low[index+1]
    is_ob_down = is_up.shift(2) & is_down.shift(1) & (close.shift(1) < low.shift(2))

    # Fair Value Gap (FVG) detection
    # isFvgUp(index) => low[index] > high[index+2]
    is_fvg_up = low > high.shift(2)
    # isFvgDown(index) => high[index] < low[index+2]
    is_fvg_down = high < low.shift(2)

    # Stacked OB + FVG conditions
    stacked_long = is_ob_up & is_fvg_up
    stacked_short = is_ob_down & is_fvg_down

    # Time window filter (Europe/London)
    ts = pd.to_datetime(df['time'], unit='s', utc=True)
    london_ts = ts.dt.tz_convert('Europe/London')
    hour = london_ts.hour
    minute = london_ts.minute

    # Morning window: 07:45 - 09:45
    morning = ((hour == 7) & (minute >= 45)) | (hour == 8) | ((hour == 9) & (minute <= 45))
    # Afternoon window: 14:45 - 16:45
    afternoon = ((hour == 14) & (minute >= 45)) | (hour == 15) | ((hour == 16) & (minute <= 45))
    in_window = morning | afternoon

    # Combine conditions
    long_cond = (stacked_long & in_window).fillna(False)
    short_cond = (stacked_short & in_window).fillna(False)

    # Build entry list
    entries = []
    trade_num = 1
    for i in df.index:
        if long_cond[i]:
            ts_entry = int(df.loc[i, 'time'])
            entry_time = datetime.fromtimestamp(ts_entry, tz=timezone.utc).isoformat()
            entry_price = float(df.loc[i, 'close'])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts_entry,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        if short_cond[i]:
            ts_entry = int(df.loc[i, 'time'])
            entry_time = datetime.fromtimestamp(ts_entry, tz=timezone.utc).isoformat()
            entry_price = float(df.loc[i, 'close'])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts_entry,
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