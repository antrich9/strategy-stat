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
    entries = []
    trade_num = 1

    # London time window check
    def is_in_london_window(ts):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        # Morning: 08:00 - 09:55
        morning_start = (hour == 8 and minute >= 0)
        morning_active = (hour == 8) or (hour == 9 and minute <= 55)
        # Afternoon: 14:00 - 16:55
        afternoon_active = (hour == 14) or (hour == 15) or (hour == 16 and minute <= 55)
        return morning_active or afternoon_active

    # Calculate shifted series for FVG detection
    low_shifted2 = df['low'].shift(2)
    high_shifted2 = df['high'].shift(2)
    low_shifted1 = df['low'].shift(1)
    high_shifted1 = df['high'].shift(1)

    # Detect FVG: bullish (low[2] >= high) or bearish (high[2] <= low)
    bullish_fvg = low_shifted2 >= df['high']
    bearish_fvg = high_shifted2 <= df['low']

    # For bullish FVG: top = low[2], bottom = high[2] (the gap zone boundaries)
    # For bearish FVG: top = high[2], bottom = low[2]
    fvg_top = np.where(bullish_fvg, low_shifted2, np.where(bearish_fvg, high_shifted2, np.nan))
    fvg_bottom = np.where(bullish_fvg, high_shifted2, np.where(bearish_fvg, low_shifted2, np.nan))
    fvg_polarity = np.where(bullish_fvg, 1, np.where(bearish_fvg, -1, 0))

    # Track active FVGs: list of [top, bottom, polarity, fvg_time]
    active_bull_fvgs = []
    active_bear_fvgs = []

    # Iterate through bars
    for i in range(len(df)):
        if i < 4:  # Need at least 4 bars back for FVG detection
            continue

        current_ts = df['time'].iloc[i]
        current_low = df['low'].iloc[i]
        current_high = df['high'].iloc[i]
        current_close = df['close'].iloc[i]

        if not is_in_london_window(current_ts):
            continue

        # Check entries into existing bullish FVGs (price drops into the FVG)
        for fvg in active_bull_fvgs[:]:  # Copy list to avoid modification during iteration
            fvg_top_val, fvg_bottom_val, fvg_time_val = fvg[0], fvg[1], fvg[3]
            # Price enters FVG: low drops below top but stays above bottom
            if current_low < fvg_top_val and current_low >= fvg_bottom_val:
                entry_price = current_close
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(current_ts),
                    'entry_time': datetime.fromtimestamp(current_ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(entry_price),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(entry_price),
                    'raw_price_b': float(entry_price)
                })
                trade_num += 1
                active_bull_fvgs.remove(fvg)

        # Check entries into existing bearish FVGs (price rises into the FVG)
        for fvg in active_bear_fvgs[:]:
            fvg_top_val, fvg_bottom_val, fvg_time_val = fvg[0], fvg[1], fvg[3]
            # Price enters FVG: high rises above bottom but stays below top
            if current_high > fvg_bottom_val and current_high <= fvg_top_val:
                entry_price = current_close
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(current_ts),
                    'entry_time': datetime.fromtimestamp(current_ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(entry_price),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(entry_price),
                    'raw_price_b': float(entry_price)
                })
                trade_num += 1
                active_bear_fvgs.remove(fvg)

        # Detect new FVG at this bar (using bar i-2 data)
        if not np.isnan(fvg_top[i]) and fvg_polarity[i] != 0:
            if fvg_polarity[i] == 1:  # Bullish FVG
                active_bull_fvgs.append([fvg_top[i], fvg_bottom[i], fvg_polarity[i], current_ts])
            else:  # Bearish FVG
                active_bear_fvgs.append([fvg_top[i], fvg_bottom[i], fvg_polarity[i], current_ts])

    return entries