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
    if df.empty or len(df) < 3:
        return []

    required_cols = ['time', 'open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        return []

    trade_num = 0
    entries = []

    bars = df['time'].values
    highs = df['high'].values
    lows = df['low'].values

    htf_period = "1D"
    res_map = {
        "1": 1, "3": 3, "5": 5, "10": 10, "15": 15, "30": 30, "45": 45, "60": 60,
        "120": 120, "180": 180, "240": 240, "D": 1440, "1D": 1440, "3D": 4320,
        "W": 10080, "1W": 10080, "M": 43200, "1M": 43200
    }
    bar_ms = res_map.get(htf_period, 1440) * 60 * 1000
    barhtf_ms = res_map.get(htf_period, 1440) * 60 * 1000

    if barhtf_ms == 0:
        return []

    shift_idx = max(1, int(round(barhtf_ms / bar_ms)))

    idx = 2
    while idx < len(df):
        cur_bar = bars[idx]
        prev_bar = bars[idx - 1]

        shifted_idx = idx - shift_idx
        if shifted_idx < 1:
            idx += 1
            continue

        prev_shifted_high = highs[shifted_idx]
        prev_shifted_low = lows[shifted_idx]
        cur_low = lows[idx]
        cur_high = highs[idx]
        prev_low = lows[idx - 1]
        prev_high = highs[idx - 1]

        if np.isnan(prev_shifted_high) or np.isnan(cur_low) or np.isnan(prev_low):
            idx += 1
            continue

        bull_fvg = (cur_bar != prev_bar) and (cur_low > prev_shifted_high) and (cur_low > prev_low)
        bear_fvg = (cur_bar != prev_bar) and (cur_high < prev_shifted_low) and (cur_high < prev_high)

        if bull_fvg:
            trade_num += 1
            ts = int(bars[idx])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(cur_low),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(cur_low),
                'raw_price_b': float(cur_low)
            })

        if bear_fvg:
            trade_num += 1
            ts = int(bars[idx])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(cur_high),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(cur_high),
                'raw_price_b': float(cur_high)
            })

        idx += 1

    return entries