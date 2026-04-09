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

    open_s = df['open']
    high_s = df['high']
    low_s = df['low']
    close_s = df['close']
    volume_s = df['volume']
    time_s = df['time']

    # ATR (Wilder RSI method)
    high_low = high_s - low_s
    high_close = (high_s - close_s.shift(1)).abs()
    low_close = (low_s - close_s.shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_20 = true_range.ewm(alpha=1/20, min_periods=19).mean()

    # Trend SMA
    loc = close_s.rolling(54).mean()
    loc2 = loc > loc.shift(1)

    # Filters (default disabled in PineScript inputs)
    atr = atr_20 / 1.5
    atrfilt = ((low_s - high_s.shift(2) > atr) | (low_s.shift(2) - high_s > atr))

    volume_sma_9 = volume_s.rolling(9).mean()
    volfilt = volume_s.shift(1) > volume_sma_9 * 1.5

    locfiltb = loc2
    locfilts = ~loc2

    # FVG conditions
    bfvg = (low_s > high_s.shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (high_s < low_s.shift(2)) & volfilt & atrfilt & locfilts

    # OB conditions
    is_up = close_s > open_s
    is_down = close_s < open_s

    ob_up = is_down.shift(1) & is_up & (close_s > high_s.shift(1))
    ob_down = is_up.shift(1) & is_down & (close_s < low_s.shift(1))

    # FVG detection
    fvg_up = low_s > high_s.shift(2)
    fvg_down = high_s < low_s.shift(2)

    # Time window check (London time: 07:45-09:45 and 14:45-16:45)
    def is_within_time_window(ts):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        # Morning window 07:45 - 09:45
        is_morning = (hour == 7 and minute >= 45) or (hour == 8) or (hour == 9 and minute < 45)
        # Afternoon window 14:45 - 16:45
        is_afternoon = (hour == 14 and minute >= 45) or (hour == 15) or (hour == 16 and minute < 45)
        return is_morning or is_afternoon

    # Build entry conditions
    long_entry = ob_up & fvg_up & bfvg
    short_entry = ob_down & fvg_down & sfvg

    # Iterate through bars
    for i in range(2, len(df)):
        if pd.isna(atr_20.iloc[i]) or pd.isna(loc.iloc[i]):
            continue

        if not is_within_time_window(time_s.iloc[i]):
            continue

        if long_entry.iloc[i]:
            entry_price = close_s.iloc[i]
            entry_ts = int(time_s.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

        if short_entry.iloc[i]:
            entry_price = close_s.iloc[i]
            entry_ts = int(time_s.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries