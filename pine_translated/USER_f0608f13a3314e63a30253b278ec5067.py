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
    # Input parameters from Pine Script
    lookback_bars = 12
    threshold = 0.0
    inp1 = False  # Volume Filter
    inp2 = False  # ATR Filter
    inp3 = False  # Trend Filter

    # Initialize result list
    entries = []
    trade_num = 1

    # Get arrays for easier handling
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    volume = df['volume'].values
    time = df['time'].values

    n = len(df)

    # Calculate indicators
    # Volume filter: volfilt = volume[1] > ta.sma(volume, 9)*1.5
    sma_9_vol = pd.Series(volume).rolling(9).mean()
    volfilt = pd.Series(volume).shift(1) > sma_9_vol.shift(1) * 1.5

    # ATR filter: atr = ta.atr(20) / 1.5, atrfilt = (low - high[2] > atr) or (low[2] - high > atr)
    def wilder_atr(high_arr, low_arr, close_arr, period=20):
        tr = np.maximum(high_arr - low_arr, 
                        np.maximum(np.abs(high_arr - np.roll(close_arr, 1)),
                                   np.abs(low_arr - np.roll(close_arr, 1))))
        tr[0] = high_arr[0] - low_arr[0]
        atr = np.zeros_like(tr)
        atr[period - 1] = np.mean(tr[:period])
        multiplier = 1.0 / period
        for i in range(period, len(tr)):
            atr[i] = atr[i - 1] * (1 - multiplier) + tr[i] * multiplier
        return atr

    atr_vals = wilder_atr(high, low, close, 20) / 1.5
    atrfilt = (low - np.roll(high, 2) > atr_vals) | (np.roll(low, 2) - high > atr_vals)
    atrfilt[0] = False
    atrfilt[1] = False

    # Trend filter: loc = ta.sma(close, 54), loc2 = loc > loc[1]
    sma_54 = pd.Series(close).rolling(54).mean()
    loc = sma_54.values
    loc2 = loc > np.roll(loc, 1)
    loc2[0] = False
    locfiltb = loc2 if inp3 else np.ones(n, dtype=bool)
    locfilts = ~loc2 if inp3 else np.ones(n, dtype=bool)

    # FVG conditions
    # bear_fvg = high < low[2] and close[1] < low[2]
    # bull_fvg = low > high[2] and close[1] > high[2]
    bear_fvg = (high < np.roll(low, 2)) & (np.roll(close, 1) < np.roll(low, 2))
    bull_fvg = (low > np.roll(high, 2)) & (np.roll(close, 1) > np.roll(high, 2))

    # bear_fvg[0] and bear_fvg[1] are invalid due to roll
    bear_fvg[0] = False
    bear_fvg[1] = False
    bull_fvg[0] = False
    bull_fvg[1] = False

    # BPR Bullish logic
    # bull_since = ta.barssince(bear_fvg)
    bull_since = np.full(n, -1, dtype=float)
    counter = 0
    for i in range(n):
        if bear_fvg[i]:
            counter = 0
        if counter >= 0:
            bull_since[i] = counter
        counter += 1

    bull_cond_1 = bear_fvg & (bull_since <= lookback_bars)
    combined_low_bull = np.where(bull_cond_1, 
                                  np.maximum(np.roll(high, 1) * np.where(np.arange(n) - bull_since >= 0, 1, 0) + high * 0, 
                                             np.roll(high, 2)), 
                                  np.nan)
    # Simplified combined_low_bull calculation
    combined_low_bull = np.full(n, np.nan)
    for i in range(n):
        if bull_cond_1[i] and bull_since[i] >= 0:
            idx = int(i - bull_since[i])
            if idx >= 0 and idx < n:
                combined_low_bull[i] = max(high[idx], high[2] if i >= 2 else low[i])

    combined_high_bull = np.full(n, np.nan)
    for i in range(n):
        if bull_cond_1[i] and bull_since[i] >= 0:
            idx2 = int(i - bull_since[i] - 2)
            if idx2 >= 0 and idx2 < n:
                combined_high_bull[i] = min(low[idx2], low[i])

    bull_result = bull_cond_1 & ~np.isnan(combined_low_bull) & ~np.isnan(combined_high_bull) & \
                  (combined_high_bull - combined_low_bull >= threshold)

    # BPR Bearish logic
    # bear_since = ta.barssince(bull_fvg)
    bear_since = np.full(n, -1, dtype=float)
    counter = 0
    for i in range(n):
        if bull_fvg[i]:
            counter = 0
        if counter >= 0:
            bear_since[i] = counter
        counter += 1

    bear_cond_1 = bull_fvg & (bear_since <= lookback_bars)
    combined_low_bear = np.full(n, np.nan)
    for i in range(n):
        if bear_cond_1[i] and bear_since[i] >= 0:
            idx = int(i - bear_since[i])
            if idx >= 0 and idx < n:
                combined_low_bear[i] = max(high[idx], high[i])

    combined_high_bear = np.full(n, np.nan)
    for i in range(n):
        if bear_cond_1[i] and bear_since[i] >= 0:
            idx2 = int(i - bear_since[i] + 2)
            if idx2 >= 0 and idx2 < n:
                combined_high_bear[i] = min(low[idx2], low[2] if i >= 2 else low[i])

    bear_result = bear_cond_1 & ~np.isnan(combined_low_bear) & ~np.isnan(combined_high_bear) & \
                  (combined_high_bear - combined_low_bear >= threshold)

    # FVG with filters (bfvg, sfvg)
    # bfvg = low > high[2] and volfilt and atrfilt and locfiltb
    # sfvg = high < low[2] and volfilt and atrfilt and locfilts
    bfvg = bull_fvg.copy()
    if inp1:
        bfvg = bfvg & volfilt.values
    if inp2:
        bfvg = bfvg & atrfilt
    if inp3:
        bfvg = bfvg & locfiltb

    sfvg = bear_fvg.copy()
    if inp1:
        sfvg = sfvg & volfilt.values
    if inp2:
        sfvg = sfvg & atrfilt
    if inp3:
        sfvg = sfvg & locfilts

    # Entry conditions: bull_result (BPR long) or bfvg (FVG long)
    # or bear_result (BPR short) or sfvg (FVG short)
    long_entry = bull_result | bfvg
    short_entry = bear_result | sfvg

    # Iterate through bars and generate entries
    for i in range(n):
        # Skip if any required indicator is NaN at this bar
        if i < 2:
            continue
        if np.isnan(loc[i]) if inp3 else False:
            continue
        if np.isnan(atr_vals[i]) if inp2 else False:
            continue

        entry_price = close[i]

        # Check for long entry
        if long_entry[i]:
            entry_time = datetime.fromtimestamp(time[i] / 1000, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(time[i]),
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1

        # Check for short entry
        if short_entry[i]:
            entry_time = datetime.fromtimestamp(time[i] / 1000, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(time[i]),
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1

    return entries