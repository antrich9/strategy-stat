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
    # Handle insufficient data
    if len(df) < 5:
        return []

    # Extract series
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    close_arr = df['close'].values
    volume_arr = df['volume'].values
    time_arr = df['time'].values

    n = len(df)

    # Helper functions for bar comparisons
    def is_up(idx):
        return close_arr[idx] > open_arr[idx]

    def is_down(idx):
        return close_arr[idx] < open_arr[idx]

    def is_ob_up(idx):
        return is_down(idx + 1) and is_up(idx) and close_arr[idx] > high_arr[idx + 1]

    def is_ob_down(idx):
        return is_up(idx + 1) and is_down(idx) and close_arr[idx] < low_arr[idx + 1]

    def is_fvg_up(idx):
        return low_arr[idx] > high_arr[idx + 2]

    def is_fvg_down(idx):
        return high_arr[idx] < low_arr[idx + 2]

    # Initialize arrays
    ob_up = np.full(n, False)
    ob_down = np.full(n, False)
    fvg_up = np.full(n, False)
    fvg_down = np.full(n, False)
    bfvg = np.full(n, False)
    sfvg = np.full(n, False)
    prev_day_high = np.full(n, np.nan)
    prev_day_low = np.full(n, np.nan)
    is_valid_time = np.full(n, False)

    # Calculate previous day high/low (simplified: rolling max/min of previous day)
    # Since we don't have daily resolution, approximate using rolling 24 periods (assuming hourly data)
    for i in range(24, n):
        prev_day_high[i] = np.max(high_arr[i-24:i])
        prev_day_low[i] = np.min(low_arr[i-24:i])

    # Calculate OB and FVG conditions
    for i in range(2, n - 2):
        ob_up[i] = is_ob_up(i - 1)
        ob_down[i] = is_ob_down(i - 1)
        fvg_up[i] = is_fvg_up(i - 1)
        fvg_down[i] = is_fvg_down(i - 1)

    # Calculate SMA for volume filter
    vol_sma = pd.Series(volume_arr).rolling(9).mean().values

    # Calculate ATR (Wilder method)
    tr = np.zeros(n)
    tr[0] = high_arr[0] - low_arr[0]
    for i in range(1, n):
        hl = high_arr[i] - low_arr[i]
        hc = abs(high_arr[i] - close_arr[i-1])
        lc = abs(low_arr[i] - close_arr[i-1])
        tr[i] = max(hl, hc, lc)

    atr = np.zeros(n)
    atr[19] = np.mean(tr[1:20])
    for i in range(20, n):
        atr[i] = (atr[i-1] * 19 + tr[i]) / 20
    atr_scaled = atr / 1.5

    # Calculate SMA for trend filter
    close_sma = pd.Series(close_arr).rolling(54).mean().values

    # Calculate time-based valid trade windows
    for i in range(n):
        dt = datetime.fromtimestamp(time_arr[i], tz=timezone.utc)
        hour = dt.hour
        is_valid_time[i] = (hour >= 2 and hour < 5) or (hour >= 10 and hour < 12)

    # Calculate FVGs with filters
    for i in range(3, n):
        if i >= 1 and i < n:
            vol_filt = volume_arr[i-1] > vol_sma[i-1] * 1.5 if not np.isnan(vol_sma[i-1]) else True
            low_diff = low_arr[i] - high_arr[i-2] if i >= 2 else 0
            high_diff = low_arr[i-2] - high_arr[i] if i >= 2 else 0
            atr_filt = (low_diff > atr_scaled[i]) or (high_diff > atr_scaled[i])

            loc = close_sma[i] if not np.isnan(close_sma[i]) else close_arr[i]
            loc_prev = close_sma[i-1] if not np.isnan(close_sma[i-1]) else close_arr[i-1]
            loc2_bull = loc > loc_prev
            loc2_bear = loc <= loc_prev

            bull_fvg_cond = low_arr[i] > high_arr[i-2]
            bear_fvg_cond = high_arr[i] < low_arr[i-2]

            bfvg[i] = bull_fvg_cond and vol_filt and atr_filt and loc2_bull
            sfvg[i] = bear_fvg_cond and vol_filt and atr_filt and loc2_bear

    # Define sweep conditions
    # PDHL Sweep + stacked OB/FVG logic
    long_entry = np.full(n, False)
    short_entry = np.full(n, False)

    for i in range(5, n):
        # Long entry: bullish FVG and sweep of prev day low (or current day low near prev day low)
        # OR stacked bullish conditions
        if bfvg[i] and ob_up[i] and is_valid_time[i]:
            long_entry[i] = True

        # Additional long: previous day low sweep with bullish confirmation
        if i >= 1 and not np.isnan(prev_day_low[i]):
            pd_low_sweep = low_arr[i] <= prev_day_low[i] and close_arr[i] > prev_day_low[i]
            if pd_low_sweep and (fvg_up[i] or ob_up[i]) and is_valid_time[i]:
                long_entry[i] = True

        # Short entry: bearish FVG and sweep of prev day high (or current day high near prev day high)
        if sfvg[i] and ob_down[i] and is_valid_time[i]:
            short_entry[i] = True

        # Additional short: previous day high sweep with bearish confirmation
        if i >= 1 and not np.isnan(prev_day_high[i]):
            pd_high_sweep = high_arr[i] >= prev_day_high[i] and close_arr[i] < prev_day_high[i]
            if pd_high_sweep and (fvg_down[i] or ob_down[i]) and is_valid_time[i]:
                short_entry[i] = True

    # Generate entry list
    entries = []
    trade_num = 1

    for i in range(n):
        direction = None
        if long_entry[i]:
            direction = 'long'
        elif short_entry[i]:
            direction = 'short'
        else:
            continue

        entry_price = close_arr[i]
        entry_ts = int(time_arr[i])
        entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()

        entry = {
            'trade_num': trade_num,
            'direction': direction,
            'entry_ts': entry_ts,
            'entry_time': entry_time,
            'entry_price_guess': float(entry_price),
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': float(entry_price),
            'raw_price_b': float(entry_price)
        }
        entries.append(entry)
        trade_num += 1

    return entries