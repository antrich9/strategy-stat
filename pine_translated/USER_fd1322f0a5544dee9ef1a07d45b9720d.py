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
    # Parameters from the original Pine Script
    fib_level_38 = 38.2
    fib_level_50 = 50.0
    fib_target = 161.8
    stop_level = 23.6
    risk_reward_ratio = 3.0

    low = df['low']
    high = df['high']

    # Swing high/low detection at offset 2 (2 bars ago)
    swing_low_2 = (low.shift(2) < low.shift(3)) & (low.shift(2) < low.shift(1))
    swing_high_2 = (high.shift(2) > high.shift(3)) & (high.shift(2) > high.shift(1))
    swing_low_2 = swing_low_2.fillna(False)
    swing_high_2 = swing_high_2.fillna(False)

    # State variables (var floats in Pine)
    pointA = np.nan
    pointB = np.nan
    pointC = np.nan
    fib_retracement = np.nan
    entry_price = np.nan
    stop_loss = np.nan
    target_price = np.nan

    entries = []
    trade_num = 0

    # Need at least 3 bars for the offset-2 swing detection
    for i in range(3, len(df)):
        # Update pointA on swing low
        if swing_low_2.iloc[i]:
            pointA = low.iloc[i - 2]

        # Update pointB on swing high (requires pointA already set)
        if swing_high_2.iloc[i] and not pd.isna(pointA):
            pointB = high.iloc[i - 2]

        # Update pointC on swing low (requires both pointA and pointB)
        if swing_low_2.iloc[i] and not pd.isna(pointA) and not pd.isna(pointB):
            pointC = low.iloc[i - 2]

            # Compute Fibonacci levels (avoid division by zero)
            if pointB != pointA:
                fib_retracement = (pointB - pointC) / (pointB - pointA) * 100.0
                entry_price = pointC + (pointB - pointC) * (fib_level_50 / 100.0)
                stop_loss = pointC + (pointB - pointC) * (stop_level / 100.0)
                target_price = pointC + (pointB - pointC) * (fib_target / 100.0)
            else:
                fib_retracement = np.nan
                entry_price = np.nan
                stop_loss = np.nan
                target_price = np.nan

            # Entry trigger conditions
            if not pd.isna(fib_retracement) and fib_retracement >= fib_level_38 and fib_retracement <= fib_level_50:
                risk = abs(entry_price - stop_loss)
                reward = abs(target_price - entry_price)
                rr_ratio = reward / risk if risk > 0 else np.inf

                if rr_ratio >= risk_reward_ratio:
                    trade_num += 1
                    entry_ts = int(df['time'].iloc[i])
                    entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                    entry_price_guess = entry_price
                    raw_price_a = raw_price_b = entry_price_guess

                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'long',
                        'entry_ts': entry_ts,
                        'entry_time': entry_time_str,
                        'entry_price_guess': entry_price_guess,
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': raw_price_a,
                        'raw_price_b': raw_price_b,
                    })

    return entries