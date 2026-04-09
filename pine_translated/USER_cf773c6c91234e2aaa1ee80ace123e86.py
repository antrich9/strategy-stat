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

    # Extract columns
    open_prices = df['open'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    close_prices = df['close'].values
    volumes = df['volume'].values
    timestamps = df['time'].values

    # Create pandas series for indicator calculations
    open_s = pd.Series(open_prices)
    high_s = pd.Series(high_prices)
    low_s = pd.Series(low_prices)
    close_s = pd.Series(close_prices)
    volume_s = pd.Series(volumes)

    n = len(df)

    # Helper functions for conditions
    def is_up(idx):
        return close_prices[idx] > open_prices[idx]

    def is_down(idx):
        return close_prices[idx] < open_prices[idx]

    def is_ob_up(idx):
        return is_down(idx + 1) and is_up(idx) and close_prices[idx] > high_prices[idx + 1]

    def is_ob_down(idx):
        return is_up(idx + 1) and is_down(idx) and close_prices[idx] < low_prices[idx + 1]

    def is_fvg_up(idx):
        return low_prices[idx] > high_prices[idx + 2]

    def is_fvg_down(idx):
        return high_prices[idx] < low_prices[idx + 2]

    # Volume filter
    vol_sma9 = volume_s.rolling(9).mean()
    volfilt = volumes > vol_sma9.values * 1.5

    # ATR filter (Wilder ATR)
    tr = pd.concat([
        high_s - low_s,
        (high_s - close_s.shift(1)).abs(),
        (low_s - close_s.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_raw = tr.ewm(alpha=1/20, adjust=False).mean()
    atr_val = atr_raw / 1.5
    atrfilt = (low_s - high_s.shift(2) > atr_val) | (low_s.shift(2) - high_s > atr_val)

    # Trend filter (SMA 54)
    loc = close_s.rolling(54).mean()
    locfiltb = loc > loc.shift(1)
    locfilts = ~locfiltb

    # Bullish and bearish FVG
    bfvg = (low_s > high_s.shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (high_s < low_s.shift(2)) & volfilt & atrfilt & locfilts

    # Stacked OB+FVG conditions
    ob_up = np.zeros(n, dtype=bool)
    ob_down = np.zeros(n, dtype=bool)
    fvg_up = np.zeros(n, dtype=bool)
    fvg_down = np.zeros(n, dtype=bool)

    for i in range(2, n - 2):
        ob_up[i] = is_ob_up(i)
        ob_down[i] = is_ob_down(i)
        fvg_up[i] = is_fvg_up(i)
        fvg_down[i] = is_fvg_down(i)

    # Get previous day high and low
    prev_day_high = np.zeros(n)
    prev_day_low = np.zeros(n)

    for i in range(n):
        current_ts = timestamps[i]
        current_dt = datetime.fromtimestamp(current_ts / 1000, tz=timezone.utc)
        prev_day_start = current_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        for j in range(i - 1, -1, -1):
            dt_j = datetime.fromtimestamp(timestamps[j] / 1000, tz=timezone.utc)
            if dt_j.date() < current_dt.date():
                prev_day_high[i] = high_prices[j]
                prev_day_low[i] = low_prices[j]
                break

    # Trading windows: 07:00-10:59 and 15:00-16:59 UTC
    in_trading_window = np.zeros(n, dtype=bool)
    for i in range(n):
        dt = datetime.fromtimestamp(timestamps[i] / 1000, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        # Window 1: 07:00-10:59
        window1 = (7 <= hour <= 10) and not (hour == 10 and minute > 59)
        # Window 2: 15:00-16:59
        window2 = (15 <= hour <= 16) and not (hour == 16 and minute > 59)
        in_trading_window[i] = window1 or window2

    # PDHL Sweep conditions
    pdhl_sweep_long = (close_s > prev_day_high) & (prev_day_high > 0)
    pdhl_sweep_short = (close_s < prev_day_low) & (prev_day_low > 0)

    # Entry conditions: stacked OB+FVG with PDHL sweep within trading window
    long_condition = ob_up & fvg_up & pdhl_sweep_long & in_trading_window
    short_condition = ob_down & fvg_down & pdhl_sweep_short & in_trading_window

    # Generate entries
    for i in range(n):
        if i < 5:
            continue
        if long_condition.iloc[i] if isinstance(long_condition, pd.Series) else long_condition[i]:
            trade_num += 1
            ts = int(timestamps[i])
            entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            entry_price = float(close_prices[i])
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })

        if short_condition.iloc[i] if isinstance(short_condition, pd.Series) else short_condition[i]:
            trade_num += 1
            ts = int(timestamps[i])
            entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            entry_price = float(close_prices[i])
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })

    return results