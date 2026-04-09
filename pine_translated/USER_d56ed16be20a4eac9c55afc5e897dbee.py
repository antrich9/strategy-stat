import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Ensure required columns exist
    required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Parameters
    PP = 5  # pivot period
    lock_bars = PP  # number of bars to skip after entry

    # Compute pivot highs and lows
    # Use a simple pivot detection: a bar is a pivot high if it's the highest in the window [i-PP:i+PP+1]
    # and also the high is unique (i.e., > neighbors)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    time = df['time'].values

    n = len(df)
    pivot_high_idx = []
    pivot_low_idx = []
    pivot_high_val = []
    pivot_low_val = []

    for i in range(PP, n - PP):
        # Check pivot high
        if high[i] == max(high[i-PP:i+PP+1]) and high[i] > high[i-1] and high[i] > high[i+1]:
            pivot_high_idx.append(i)
            pivot_high_val.append(high[i])
        # Check pivot low
        if low[i] == min(low[i-PP:i+PP+1]) and low[i] < low[i-1] and low[i] < low[i+1]:
            pivot_low_idx.append(i)
            pivot_low_val.append(low[i])

    # Build lists to track major highs/lows
    # We'll just use the last two pivot highs/lows as major
    major_high_idx = []
    major_low_idx = []
    major_high_val = []
    major_low_val = []

    # We need to track the sequence of pivots
    # We'll combine pivot highs and lows into a single timeline sorted by index
    # We'll assign types: 'H' for high, 'L' for low
    pivots = []
    for idx, val in zip(pivot_high_idx, pivot_high_val):
        pivots.append((idx, val, 'H'))
    for idx, val in zip(pivot_low_idx, pivot_low_val):
        pivots.append((idx, val, 'L'))
    # Sort by index
    pivots.sort(key=lambda x: x[0])

    # Now iterate through pivots to detect BoS and ChoCh
    # We'll track the last few pivots
    # Use a lock to avoid repeated entries
    lock_counter = 0
    last_entry_idx = -1

    entries = []
    trade_num = 1

    # We'll track the last major high and low
    # Use a simple approach: the last pivot high is major high, last pivot low is major low
    # But we need to differentiate between major and minor. We'll just use the most recent pivot high and low as major.
    # However, we need to detect BoS: break of structure occurs when price breaks above/below the previous major high/low after a pullback.
    # We'll track the previous major high/low and the one before that.

    # We'll store the last two pivot highs and lows
    last_highs = []  # list of (idx, val)
    last_lows = []   # list of (idx, val)

    for i, (idx, val, typ) in enumerate(pivots):
        if typ == 'H':
            last_highs.append((idx, val))
            # Keep only last 3
            if len(last_highs) > 3:
                last_highs.pop(0)
        else:
            last_lows.append((idx, val))
            if len(last_lows) > 3:
                last_lows.pop(0)

        # Need at least 2 highs and 2 lows to detect BoS
        if len(last_highs) >= 2 and len(last_lows) >= 2:
            # Major high is the most recent high, major low is the most recent low
            major_high = last_highs[-1][1]
            major_low = last_lows[-1][1]
            # Previous major high and low
            prev_major_high = last_highs[-2][1]
            prev_major_low = last_lows[-2][1]

            # Detect bullish BoS: price breaks above previous major high and previous major low is lower
            # Use close price at current bar (idx)
            # But we need to check if close crosses above prev major high
            # Use crossover function: close crosses above prev major high
            # For bullish BoS: close > prev_major_high and close > major_low (maybe?), and previous low is lower
            # Actually typical BoS: price breaks above the high of the previous swing (prev major high) after a pullback (i.e., price made a lower low)
            # So we need to check if close > prev_major_high and prev_major_low < major_low (i.e., a lower low)
            # But we also need to ensure that the pullback was a lower low (i.e., major low is lower than prev major low)
            # However, we can simplify: if close > prev_major_high and major_low < prev_major_low, then bullish BoS

            # Similarly, bearish BoS: close < prev_major_low and major_high > prev_major_high

            # For ChoCh: change in character: e.g., after a series of lower highs, a higher high appears
            # Detect: last_highs[-1] > last_highs[-2] and last_highs[-2] < last_highs[-3] (i.e., a higher high after lower highs)
            # This is a bullish ChoCh

            # We'll generate entries for BoS and ChoCh

            # For simplicity, we generate entry on the bar after the pivot is confirmed (i.e., at idx)
            # But we need to check the condition at the bar after the pivot? In Pine, the pivot is detected at the bar when the condition is met (the current bar). The entry would be on the next bar? Usually entry is at the close of the bar when the condition is met, or at the open of the next bar.

            # We'll use the close of the current bar (idx) to check the condition, and then generate entry at that same bar's close price.

            # However, we need to check the condition using the previous bar's values. For crossover detection, we need to compare the current close with the previous major high and the previous close.

            # Let's compute close at idx and previous close at idx-1
            if idx > 0:
                prev_close = close[idx-1]
                curr_close = close[idx]
                # For bullish BoS: curr_close > prev_major_high and prev_close <= prev_major_high
                if curr_close > prev_major_high and prev_close <= prev_major_high:
                    # Check if we are not in lock period
                    if idx - last_entry_idx >= lock_bars:
                        # Long entry
                        entry_time = datetime.fromtimestamp(time[idx], tz=timezone.utc).isoformat()
                        entries.append({
                            'trade_num': trade_num,
                            'direction': 'long',
                            'entry_ts': int(time[idx]),
                            'entry_time': entry_time,
                            'entry_price_guess': float(curr_close),
                            'exit_ts': 0,
                            'exit_time': '',
                            'exit_price_guess': 0.0,
                            'raw_price_a': float(curr_close),
                            'raw_price_b': float(curr_close)
                        })
                        trade_num += 1
                        last_entry_idx = idx
                        lock_counter = lock_bars

                # For bearish BoS: curr_close < prev_major_low and prev_close >= prev_major_low
                if curr_close < prev_major_low and prev_close >= prev_major_low:
                    if idx - last_entry_idx >= lock_bars:
                        # Short entry
                        entry_time = datetime.fromtimestamp(time[idx], tz=timezone.utc).isoformat()
                        entries.append({
                            'trade_num': trade_num,
                            'direction': 'short',
                            'entry_ts': int(time[idx]),
                            'entry_time': entry_time,
                            'entry_price_guess': float(curr_close),
                            'exit_ts': 0,
                            'exit_time': '',
                            'exit_price_guess': 0.0,
                            'raw_price_a': float(curr_close),
                            'raw_price_b': float(curr_close)
                        })
                        trade_num += 1
                        last_entry_idx = idx
                        lock_counter = lock_bars

            # For ChoCh detection: need at least 3 highs
            if len(last_highs) >= 3:
                # Bullish ChoCh: higher high after lower highs
                # Check if last_highs[-1] > last_highs[-2] and last_highs[-2] < last_highs[-3]
                if last_highs[-1][1] > last_highs[-2][1] and last_highs[-2][1] < last_highs[-3][1]:
                    # Check crossover: price crosses above last_highs[-1] maybe?
                    # Use close > last_highs[-1][1] and prev_close <= last_highs[-1][1]
                    if idx > 0:
                        if curr_close > last_highs[-1][1] and prev_close <= last_highs[-1][1]:
                            if idx - last_entry_idx >= lock_bars:
                                # Long entry
                                entry_time = datetime.fromtimestamp(time[idx], tz=timezone.utc).isoformat()
                                entries.append({
                                    'trade_num': trade_num,
                                    'direction': 'long',
                                    'entry_ts': int(time[idx]),
                                    'entry_time': entry_time,
                                    'entry_price_guess': float(curr_close),
                                    'exit_ts': 0,
                                    'exit_time': '',
                                    'exit_price_guess': 0.0,
                                    'raw_price_a': float(curr_close),
                                    'raw_price_b': float(curr_close)
                                })
                                trade_num += 1
                                last_entry_idx = idx
                                lock_counter = lock_bars

                # Bearish ChoCh: lower low after higher lows
                if len(last_lows) >= 3:
                    if last_lows[-1][1] < last_lows[-2][1] and last_lows[-2][1] > last_lows[-3][1]:
                        if curr_close < last_lows[-1][1] and prev_close >= last_lows[-1][1]:
                            if idx - last_entry_idx >= lock_bars:
                                # Short entry
                                entry_time = datetime.fromtimestamp(time[idx], tz=timezone.utc).isoformat()
                                entries.append({
                                    'trade_num': trade_num,
                                    'direction': 'short',
                                    'entry_ts': int(time[idx]),
                                    'entry_time': entry_time,
                                    'entry_price_guess': float(curr_close),
                                    'exit_ts': 0,
                                    'exit_time': '',
                                    'exit_price_guess': 0.0,
                                    'raw_price_a': float(curr_close),
                                    'raw_price_b': float(curr_close)
                                })
                                trade_num += 1
                                last_entry_idx = idx
                                lock_counter = lock_bars

    return entries