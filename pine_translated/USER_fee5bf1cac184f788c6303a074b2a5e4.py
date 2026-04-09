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
    open_col = df['open'].values
    high_col = df['high'].values
    low_col = df['low'].values
    close_col = df['close'].values
    volume_col = df['volume'].values
    time_col = df['time'].values
    n = len(df)

    # Extract hour from unix timestamp
    hours = np.array([datetime.fromtimestamp(ts, tz=timezone.utc).hour for ts in time_col])

    # Time filter: (hour >= 2 and hour < 5) or (hour >= 10 and hour < 12)
    isValidTradeTime = ((hours >= 2) & (hours < 5)) | ((hours >= 10) & (hours < 12))

    # Helper functions for OB and FVG
    def is_up(idx):
        return close_col[idx] > open_col[idx]

    def is_down(idx):
        return close_col[idx] < open_col[idx]

    def is_ob_up(idx):
        return is_down(idx + 1) and is_up(idx) and close_col[idx] > high_col[idx + 1]

    def is_ob_down(idx):
        return is_up(idx + 1) and is_down(idx) and close_col[idx] < low_col[idx + 1]

    def is_fvg_up(idx):
        return low_col[idx] > high_col[idx + 2]

    def is_fvg_down(idx):
        return high_col[idx] < low_col[idx + 2]

    # Previous day high and low (D timeframe)
    prev_day_high = np.full(n, np.nan)
    prev_day_low = np.full(n, np.nan)

    # Calculate D high/low manually
    for i in range(1, n):
        ts = time_col[i]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        day_start_ts = int(datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc).timestamp())
        day_end_ts = int(datetime(dt.year, dt.month, dt.day, 23, 59, 59, tzinfo=timezone.utc).timestamp())
        day_mask = (time_col >= day_start_ts) & (time_col <= day_end_ts)
        if np.any(day_mask):
            prev_day_high[i] = np.nanmax(high_col[day_mask])
            prev_day_low[i] = np.nanmin(low_col[day_mask])

    # Current 240min high/low
    current_240_high = np.full(n, np.nan)
    current_240_low = np.full(n, np.nan)

    for i in range(n):
        ts = time_col[i]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        minute_block = (dt.hour * 60 + dt.minute) // 240
        block_start = int(datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc).replace(hour=minute_block * 4 // 60, minute=(minute_block * 4) % 60).timestamp())
        block_end = block_start + 14400
        block_mask = (time_col >= block_start) & (time_col < block_end)
        if np.any(block_mask):
            current_240_high[i] = np.nanmax(high_col[block_mask])
            current_240_low[i] = np.nanmin(low_col[block_mask])

    # Volume filter
    vol_sma = pd.Series(volume_col).rolling(9).mean().values
    volfilt = volume_col > vol_sma * 1.5

    # ATR filter (Wilder ATR with period 20, divided by 1.5)
    atr = np.full(n, np.nan)
    if n > 1:
        tr = np.maximum(high_col[1:] - low_col[1:], np.maximum(np.abs(high_col[1:] - close_col[:-1]), np.abs(low_col[1:] - close_col[:-1])))
        atr[1] = tr[0]
        alpha = 1.0 / 20.0
        for i in range(2, n):
            atr[i] = alpha * tr[i-1] + (1 - alpha) * atr[i-1]
    atr_scaled = atr / 1.5

    # Low vs high[2] and low[2] vs high comparison for ATR filter
    low_vs_high_2 = low_col - np.roll(high_col, 2)
    low_2_vs_high = np.roll(low_col, 2) - high_col

    atrfilt = (low_vs_high_2 > atr_scaled) | (low_2_vs_high > atr_scaled)
    atrfilt[0] = atrfilt[1] if n > 1 else True
    atrfilt[1] = atrfilt[1] if n > 1 else True

    # Trend filter
    loc = pd.Series(close_col).ewm(span=54, adjust=False).mean().values
    loc2 = loc > np.roll(loc, 1)
    loc2[0] = False
    locfiltb = loc2
    locfilts = ~loc2

    # OB conditions
    ob_up = np.full(n, False)
    ob_down = np.full(n, False)
    for i in range(1, n - 2):
        if not np.isnan(close_col[i]):
            ob_up[i] = is_ob_up(1)
            ob_down[i] = is_ob_down(1)

    # FVG conditions
    fvg_up = np.full(n, False)
    fvg_down = np.full(n, False)
    for i in range(2, n):
        if not np.isnan(close_col[i]):
            fvg_up[i] = is_fvg_up(0)
            fvg_down[i] = is_fvg_down(0)
        fvg_up[i] = low_col[i] > high_col[i + 2] if i + 2 < n else False
        fvg_down[i] = high_col[i] < low_col[i + 2] if i + 2 < n else False

    # Reset fvg arrays properly
    fvg_up = np.full(n, False)
    fvg_down = np.full(n, False)
    for i in range(n):
        if i + 2 < n:
            fvg_up[i] = low_col[i] > high_col[i + 2]
            fvg_down[i] = high_col[i] < low_col[i + 2]

    # Main entry conditions
    bfvg = fvg_up & volfilt & atrfilt & locfiltb
    sfvg = fvg_down & volfilt & atrfilt & locfilts

    # PDHL sweep conditions
    long_sweep = (high_col > prev_day_high) & (prev_day_high > 0)
    short_sweep = (low_col < prev_day_low) & (prev_day_low > 0)

    # Combined long and short entry conditions
    long_entry = bfvg & isValidTradeTime & long_sweep
    short_entry = sfvg & isValidTradeTime & short_sweep

    # Generate entries
    entries = []
    trade_num = 1

    for i in range(2, n):
        if np.isnan(close_col[i]) or np.isnan(prev_day_high[i]) or np.isnan(prev_day_low[i]):
            continue

        direction = None
        if long_entry[i]:
            direction = 'long'
        elif short_entry[i]:
            direction = 'short'

        if direction:
            ts = int(time_col[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(close_col[i])

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