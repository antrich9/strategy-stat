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
    # Validate columns
    required = {'time', 'open', 'high', 'low', 'close', 'volume'}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    # Ensure sorted by time
    df = df.sort_values('time').reset_index(drop=True)

    # Parameters (default values matching Pine script defaults)
    thresholdPer = 0.0          # input.float default 0
    auto = False                # input(false)
    threshold_value = thresholdPer / 100.0

    # Compute auto threshold if needed
    if auto:
        ret = (df['high'] - df['low']) / df['low']
        cum_ret = ret.cumsum()
        bar_index = np.arange(len(df)) + 1
        threshold_series = cum_ret / bar_index
    else:
        threshold_series = pd.Series(threshold_value, index=df.index)

    # Shift required for FVG detection
    high_shift2 = df['high'].shift(2)
    low_shift2 = df['low'].shift(2)
    close_shift1 = df['close'].shift(1)

    # Bullish FVG: low > high[2] and close[1] > high[2] and (low - high[2]) / high[2] > threshold
    bull_fvg = (
        (df['low'] > high_shift2) &
        (close_shift1 > high_shift2) &
        ((df['low'] - high_shift2) / high_shift2 > threshold_series)
    )

    # Bearish FVG: high < low[2] and close[1] < low[2] and (low[2] - high) / low[2] > threshold
    bear_fvg = (
        (df['high'] < low_shift2) &
        (close_shift1 < low_shift2) &
        ((low_shift2 - df['high']) / low_shift2 > threshold_series)
    )

    # Build entry list
    entries = []
    trade_num = 1
    for i in range(2, len(df)):
        if bull_fvg.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
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
            trade_num += 1
        elif bear_fvg.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
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
            trade_num += 1

    return entries