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
    lookback_bars = 12
    threshold = 0.0

    n = len(df)
    entries = []
    trade_num = 1

    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    time = df['time'].values

    bear_fvg = np.zeros(n, dtype=bool)
    bull_fvg = np.zeros(n, dtype=bool)

    for i in range(2, n):
        bear_fvg[i] = high[i] < low[i-2] and close[i-1] < low[i-2]
        bull_fvg[i] = low[i] > high[i-2] and close[i-1] > high[i-2]

    bull_since = np.full(n, -1, dtype=float)
    bear_since = np.full(n, -1, dtype=float)

    last_bear_idx = -1
    last_bull_idx = -1

    for i in range(n):
        if bear_fvg[i]:
            last_bear_idx = i
        if bull_fvg[i]:
            last_bull_idx = i
        if last_bear_idx >= 0:
            bull_since[i] = i - last_bear_idx
        if last_bull_idx >= 0:
            bear_since[i] = i - last_bull_idx

    bull_cond_1 = bull_fvg & (bull_since <= lookback_bars) & (bull_since >= 0)
    bear_cond_1 = bear_fvg & (bear_since <= lookback_bars) & (bear_since >= 0)

    bull_result = np.zeros(n, dtype=bool)
    bear_result = np.zeros(n, dtype=bool)

    combined_high_bull = np.full(n, np.nan)
    combined_low_bull = np.full(n, np.nan)
    combined_high_bear = np.full(n, np.nan)
    combined_low_bear = np.full(n, np.nan)

    for i in range(2, n):
        if bull_cond_1[i]:
            bs = int(bull_since[i])
            combined_low_bull[i] = max(high[i - bs], high[i - 2])
            combined_high_bull[i] = min(low[i - bs + 2], low[i])
            if not np.isnan(combined_high_bull[i]) and not np.isnan(combined_low_bull[i]):
                if combined_high_bull[i] - combined_low_bull[i] >= threshold:
                    bull_result[i] = True

        if bear_cond_1[i]:
            bes = int(bear_since[i])
            combined_low_bear[i] = max(high[i - bes + 2], high[i])
            combined_high_bear[i] = min(low[i - bes], low[i - 2])
            if not np.isnan(combined_high_bear[i]) and not np.isnan(combined_low_bear[i]):
                if combined_high_bear[i] - combined_low_bear[i] >= threshold:
                    bear_result[i] = True

    last_entry_type = None

    for i in range(n):
        if bull_result[i]:
            if last_entry_type == "Entered Bullish FVG":
                entry_ts = int(time[i])
                entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                entry_price = float(close[i])

                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': entry_ts,
                    'entry_time': entry_time_str,
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1

            last_entry_type = "Entered Bullish FVG"

        if bear_result[i]:
            last_entry_type = "Entered Bearish FVG"

    return entries