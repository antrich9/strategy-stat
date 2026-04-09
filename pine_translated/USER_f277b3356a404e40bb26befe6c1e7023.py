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
    trade_num = 0

    open_series = df['open']
    high_series = df['high']
    low_series = df['low']
    close_series = df['close']
    volume_series = df['volume']
    time_series = df['time']

    # Helper functions for OB and FVG detection
    def is_up(idx):
        return close_series.iloc[idx] > open_series.iloc[idx]

    def is_down(idx):
        return close_series.iloc[idx] < open_series.iloc[idx]

    def is_ob_up(idx):
        return (is_down(idx - 1) and is_up(idx) and
                close_series.iloc[idx] > high_series.iloc[idx - 1])

    def is_ob_down(idx):
        return (is_up(idx - 1) and is_down(idx) and
                close_series.iloc[idx] < low_series.iloc[idx - 1])

    def is_fvg_up(idx):
        return low_series.iloc[idx] > high_series.iloc[idx + 2]

    def is_fvg_down(idx):
        return high_series.iloc[idx] < low_series.iloc[idx + 2]

    # Build time-based trading window series
    in_trading_window = pd.Series(False, index=df.index)

    for i in range(len(df)):
        ts = time_series.iloc[i]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute

        in_window_1 = (hour >= 7 and hour <= 10) and not (hour == 10 and minute > 59)
        in_window_2 = (hour >= 15 and hour <= 16) and not (hour == 16 and minute > 59)

        in_trading_window.iloc[i] = in_window_1 or in_window_2

    # Volume filter
    vol_sma = volume_series.rolling(9).mean()
    volfilt = volume_series.shift(1) > vol_sma * 1.5

    # ATR filter (Wilder smoothing)
    tr = pd.concat([
        high_series - low_series,
        (high_series - close_series.shift(1)).abs(),
        (low_series - close_series.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/20, adjust=False).mean()
    atrfilt = ((low_series - high_series.shift(2) > atr / 1.5) |
               (low_series.shift(2) - high_series > atr / 1.5))

    # Trend filter
    loc = close_series.ewm(span=54, adjust=False).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2

    # FVG conditions
    bfvg = (low_series > high_series.shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (high_series < low_series.shift(2)) & volfilt & atrfilt & locfilts

    # Stacked OB + FVG conditions
    stacked_bull = pd.Series(False, index=df.index)
    stacked_bear = pd.Series(False, index=df.index)

    for i in range(2, len(df)):
        if is_ob_up(i - 1) and is_fvg_up(i):
            stacked_bull.iloc[i] = True
        if is_ob_down(i - 1) and is_fvg_down(i):
            stacked_bear.iloc[i] = True

    # Entry signal conditions (Bull and Bear combined)
    bull_entry = stacked_bull & in_trading_window
    bear_entry = stacked_bear & in_trading_window

    for i in range(2, len(df)):
        if (bull_entry.iloc[i] and not bull_entry.iloc[i - 1] and
            not pd.isna(close_series.iloc[i]) and not pd.isna(atr.iloc[i])):
            trade_num += 1
            entry_ts = int(time_series.iloc[i])
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close_series.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close_series.iloc[i]),
                'raw_price_b': float(close_series.iloc[i])
            })
        elif (bear_entry.iloc[i] and not bear_entry.iloc[i - 1] and
              not pd.isna(close_series.iloc[i]) and not pd.isna(atr.iloc[i])):
            trade_num += 1
            entry_ts = int(time_series.iloc[i])
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close_series.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close_series.iloc[i]),
                'raw_price_b': float(close_series.iloc[i])
            })

    return results