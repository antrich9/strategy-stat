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
    pp = 5
    results = []
    trade_num = 1

    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    time = df['time'].values
    n = len(df)

    # Calculate pivots
    pivot_high = np.zeros(n)
    pivot_low = np.zeros(n)
    pivot_high_idx = np.zeros(n, dtype=int) * np.nan
    pivot_low_idx = np.zeros(n, dtype=int) * np.nan

    for i in range(pp, n - pp):
        is_high = True
        is_low = True
        for j in range(1, pp + 1):
            if high[i] <= high[i - j]:
                is_high = False
            if low[i] >= low[i - j]:
                is_low = False
        if is_high:
            for j in range(1, pp + 1):
                if high[i] < high[i + j]:
                    is_high = False
                    break
        if is_low:
            for j in range(1, pp + 1):
                if low[i] > low[i + j]:
                    is_low = False
                    break
        if is_high:
            pivot_high[i] = high[i]
            pivot_high_idx[i] = i
        if is_low:
            pivot_low[i] = low[i]
            pivot_low_idx[i] = i

    # Track major structure
    major_highs = []
    major_lows = []
    major_high_idx = []
    major_low_idx = []
    major_types = []

    last_pivot_type = None
    last_pivot_idx = 0
    last_pivot_val = 0

    for i in range(n):
        if not np.isnan(pivot_high_idx[i]):
            if last_pivot_type is None or last_pivot_type == 'L':
                major_highs.append(high[i])
                major_high_idx.append(i)
                major_types.append('H')
                last_pivot_type = 'H'
                last_pivot_idx = i
                last_pivot_val = high[i]
            elif last_pivot_type == 'H' and high[i] > last_pivot_val:
                major_highs[-1] = high[i]
                major_high_idx[-1] = i
                last_pivot_val = high[i]

        if not np.isnan(pivot_low_idx[i]):
            if last_pivot_type is None or last_pivot_type == 'H':
                major_lows.append(low[i])
                major_low_idx.append(i)
                major_types.append('L')
                last_pivot_type = 'L'
                last_pivot_idx = i
                last_pivot_val = low[i]
            elif last_pivot_type == 'L' and low[i] < last_pivot_val:
                major_lows[-1] = low[i]
                major_low_idx[-1] = i
                last_pivot_val = low[i]

    # Detect structure and generate entries
    bullish_bos = np.zeros(n, dtype=bool)
    bearish_bos = np.zeros(n, dtype=bool)
    bullish_choch = np.zeros(n, dtype=bool)
    bearish_choch = np.zeros(n, dtype=bool)

    for i in range(pp, n):
        if len(major_highs) >= 2 and len(major_lows) >= 2:
            # Bullish BoS: price breaks above previous major high
            if len(major_high_idx) >= 1 and major_high_idx[-1] < i:
                last_high_idx = major_high_idx[-1]
                if close[i] > high[last_high_idx] if last_high_idx < n else False:
                    if last_high_idx >= 0 and last_high_idx < n:
                        bullish_bos[i] = True

            # Bearish BoS: price breaks below previous major low
            if len(major_low_idx) >= 1 and major_low_idx[-1] < i:
                last_low_idx = major_low_idx[-1]
                if last_low_idx >= 0 and last_low_idx < n:
                    if close[i] < low[last_low_idx]:
                        bearish_bos[i] = True

            # Detect ChoCh patterns (change in structure)
            if len(major_types) >= 4:
                # Bullish ChoCh: previous structure was bearish, now bullish
                if len(major_types) >= 2:
                    if major_types[-2] == 'H' and major_types[-1] == 'L':
                        if close[i] > major_highs[-2] if len(major_highs) >= 2 else False:
                            bullish_choch[i] = True
                # Bearish ChoCh: previous structure was bullish, now bearish
                if major_types[-2] == 'L' and major_types[-1] == 'H':
                    if len(major_lows) >= 2:
                        if close[i] < major_lows[-2]:
                            bearish_choch[i] = True

    # Generate entry signals based on structure breaks
    long_signal = bullish_bos | bullish_choch
    short_signal = bearish_bos | bearish_choch

    # Filter signals with valid pivots
    valid_long = np.zeros(n, dtype=bool)
    valid_short = np.zeros(n, dtype=bool)

    for i in range(pp, n):
        if long_signal[i] and not np.isnan(pivot_low_idx[i]):
            valid_long[i] = True
        if short_signal[i] and not np.isnan(pivot_high_idx[i]):
            valid_short[i] = True

    # Create entries
    for i in range(pp, n):
        if valid_long[i]:
            entry_price = close[i]
            ts = int(time[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1

        if valid_short[i]:
            entry_price = close[i]
            ts = int(time[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1

    return results